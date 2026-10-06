#!/usr/bin/env python3
"""Event-level (operational) evaluation of finished checkpoints (S8f).

For every finished task under --results_dir the test split is scored once at
the task's validation-selected threshold and summarised per conflict EVENT
(see skyflow/experiments/events.py): event detection rate, timely detection
rate (lead >= --lead_s), lead time at first alert, alert-episode precision
and false alert episodes per UAV-hour.  Same code for every method.

    python scripts/eval_events.py --results_dir results/main --out_dir results

Outputs (all with provenance in events.json):
  results/events.csv            one row per method x seed
  results/events_by_cause.csv   per method x seed x conflict cause
  results/events_summary.csv    mean/std over seeds + paired t-test vs --reference (Bonferroni)
  results/events_lead.npz       lead-time samples per method/seed (for figures)
"""

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from skyflow.data.cache import cache_key, get_split
from skyflow.experiments.env_info import env_info
from skyflow.experiments.events import evaluate_events
from skyflow.experiments.loader import list_tasks, load_task
from skyflow.experiments.stats import bonferroni, paired_ttest

SUMMARY_METRICS = ("event_cdr", "timely_cdr", "lead_mean_s", "lead_median_s", "lead_frac", "episode_precision",
                   "false_episodes_per_uav_hour", "false_episode_dur_mean_s")


def write_csv(path: Path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def summarise(rows, reference):
    """Mean/std over seeds per (method, persistence) and paired t-tests vs the
    reference at the same persistence (Bonferroni over methods)."""
    out, tests = [], {}
    for M in sorted({int(r["persistence"]) for r in rows}):
        sub = [r for r in rows if int(r["persistence"]) == M]
        o, t = _summarise_one(sub, reference)
        for rec in o:
            rec["persistence"] = M
        out += o
        tests[f"persistence_{M}"] = t
    return out, tests


def _summarise_one(rows, reference):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out, tests = [], {}
    ref_rows = {r["seed"]: r for r in by_method.get(reference, [])}
    for m, rs in by_method.items():
        rec = {"method": m, "kind": rs[0]["kind"], "n_seeds": len(rs), "n_events": rs[0]["n_events"],
               "uav_hours": rs[0]["uav_hours"]}
        for k in SUMMARY_METRICS:
            v = np.array([float(r[k]) for r in rs], dtype=float)
            rec[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            rec[f"{k}_std"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
        out.append(rec)
    for met in ("event_cdr", "timely_cdr", "false_episodes_per_uav_hour", "episode_precision"):
        raw = {}
        for m, rs in by_method.items():
            if m == reference:
                continue
            seeds = sorted(set(r["seed"] for r in rs) & set(ref_rows))
            if len(seeds) < 2:
                continue
            a = [float(ref_rows[s][met]) for s in seeds]
            b = [float(next(r for r in rs if r["seed"] == s)[met]) for s in seeds]
            res = paired_ttest(a, b)
            res["seeds"] = seeds
            raw[m] = res
        corr = bonferroni({m: r["p"] for m, r in raw.items()})
        for m in raw:
            raw[m]["p_bonferroni"] = corr[m]
        tests[f"paired_t_{met}"] = raw
        for rec in out:
            if rec["method"] in raw:
                rec[f"p_{met}_bonf"] = raw[rec["method"]]["p_bonferroni"]
                rec[f"d_{met}"] = raw[rec["method"]]["mean_diff"]       # reference minus method
    return out, tests


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--lead_s", type=float, default=10.0, help="lead time (s) that counts as timely")
    ap.add_argument("--persistence", nargs="+", type=int, default=[1, 3],
                    help="M-of-M alert persistence filters to report (1 = raw alerts)")
    ap.add_argument("--reference", default="TR-GAT")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _smoke)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = [(m, s, d) for m, s, d in list_tasks(args.results_dir)
             if (args.methods is None or m in args.methods) and (args.seeds is None or s in args.seeds)]
    if not tasks:
        raise SystemExit(f"no finished tasks under {args.results_dir}")
    print(f"{len(tasks)} task(s) on {device}; lead_s={args.lead_s}")

    t_start = time.perf_counter()
    rows, cause_rows, leads, data_cache = [], [], {}, {}
    for method, seed, d in tasks:
        t0 = time.perf_counter()
        lm = load_task(d, device)
        cfg = lm.cfg
        if lm.kind == "rule" and hasattr(lm.model, "fit") and "rule_config" not in (lm.metrics or {}).get("training", {}):
            lm.model.fit(get_split(cfg, "val", cache_dir=args.cache_dir, verbose=False))
        key = cache_key(cfg, args.split)
        if key not in data_cache:
            data_cache[key] = get_split(cfg, args.split, cache_dir=args.cache_dir, verbose=False)
        data = data_cache[key]
        n_scen = cfg.data.split_scenarios(args.split)
        if len(data) % n_scen != 0:
            raise SystemExit(f"{len(data)} snapshots not divisible into {n_scen} scenarios")
        sps = len(data) // n_scen
        dt = float(cfg.data.scenario_duration_s) / sps
        results = evaluate_events(lm, data, snapshots_per_scenario=sps, lead_s=args.lead_s, snapshot_dt_s=dt,
                                  persistence=args.persistence)
        for res in results:
            row = {"method": method, "kind": lm.kind, "seed": seed,
                   "threshold": lm.trainer.threshold if lm.kind == "trgat" else getattr(lm.model, "threshold", None),
                   "checkpoint_epoch": lm.checkpoint_epoch, "snapshots": len(data), "scenarios": n_scen,
                   "snapshot_dt_s": dt, "lead_s": args.lead_s}
            row.update(res.to_row())
            rows.append(row)
            for cname, cv in res.per_cause.items():
                cause_rows.append({"method": method, "seed": seed, "persistence": res.persistence, "cause": cname, **cv})
            leads[f"{method}/seed{seed}/p{res.persistence}"] = res.lead_s_list
            print(f"{method:18s} seed={seed} M={res.persistence}: events={res.n_events} event-CDR={res.event_cdr:.3f} "
                  f"timely={res.timely_cdr:.3f} lead med={res.lead_s['median']:.1f}s | episodes={res.n_alert_episodes} "
                  f"prec={res.episode_precision:.3f} false/UAV-h={res.false_episodes_per_uav_hour:.2f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    O = Path(args.out_dir)
    fields = list(rows[0].keys())
    write_csv(O / f"events{args.tag}.csv", rows, fields)
    if cause_rows:
        write_csv(O / f"events_by_cause{args.tag}.csv", cause_rows, list(cause_rows[0].keys()))
    summary, tests = summarise(rows, args.reference)
    sfields = sorted({k for r in summary for k in r},
                     key=lambda k: (k not in ("method", "kind", "persistence", "n_seeds"), k))
    write_csv(O / f"events_summary{args.tag}.csv", summary, sfields)
    np.savez_compressed(O / f"events_lead{args.tag}.npz", **{k: v for k, v in leads.items() if v is not None})
    meta = {"results_dir": args.results_dir, "split": args.split, "lead_s": args.lead_s, "reference": args.reference,
            "tasks": [str(d) for _, _, d in tasks], "tests": tests, "env": env_info(device),
            "seconds": time.perf_counter() - t_start, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(O / f"events{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"written: {O / f'events{args.tag}.csv'}, events_by_cause, events_summary, events_lead.npz, events.json")


if __name__ == "__main__":
    main()
