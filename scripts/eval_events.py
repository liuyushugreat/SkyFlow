#!/usr/bin/env python3
"""Event-level (operational) evaluation of finished checkpoints (S8f).

For every finished task under --results_dir the validation and test splits
are scored once; the per-snapshot scores are then summarised per conflict
EVENT (see skyflow/experiments/events.py) in three ways, identically for
every method:

  1. raw operating point   validation-F1 threshold (as in the main table),
                           plus the same with a 3-of-3 persistence filter
  2. SOC curve             System Operating Characteristic (Kuchar 1996): event
                           CDR vs false alert episodes per UAV-hour over a
                           threshold grid, for each operational-layer setting
                           (EMA alpha x hysteresis) -- test split
  3. budget-matched point  on the VALIDATION split, the (alpha, hysteresis,
                           threshold) with the highest event CDR whose false
                           episode rate is within a budget; evaluated on test.
                           Budgets: fixed values (--budgets) and the validation
                           false-episode rate of a reference rule (--match)

    python scripts/eval_events.py --results_dir results/main --out_dir results

Outputs (provenance in events.json):
  events.csv / events_by_cause.csv / events_summary.csv / events_lead.npz   (1)
  events_soc.csv                                                            (2)
  events_budget.csv / events_budget_summary.csv                             (3)
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
from skyflow.experiments.events import (event_metrics, filtered_alerts, score_table, select_operating_point,
                                        soc_curve, threshold_grid)
from skyflow.experiments.loader import list_tasks, load_task
from skyflow.experiments.methods import METHODS
from skyflow.experiments.stats import bonferroni, paired_ttest

SUMMARY_METRICS = ("event_cdr", "timely_cdr", "lead_mean_s", "lead_median_s", "lead_frac", "episode_precision",
                   "false_episodes_per_uav_hour", "false_episode_dur_mean_s")
TEST_METRICS = ("event_cdr", "timely_cdr", "false_episodes_per_uav_hour", "episode_precision")
SOC_METHODS = ["TR-GAT", "TR-GAT-NT", "GAT-S", "STGCN", "LSTM-P", "Tfm-P", "CPA-Rule", "Plan-CPA", "VO"]


def write_csv(path: Path, rows, fields=None):
    if not rows:
        return
    fields = fields or list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})


def summarise(rows, reference, group_key, metrics=SUMMARY_METRICS, tests_on=TEST_METRICS):
    """Mean/std over seeds per (method, group_key) and paired t-tests vs the
    reference within the same group (Bonferroni over methods)."""
    out, tests = [], {}
    for G in sorted({r[group_key] for r in rows}, key=lambda x: float(x) if _isnum(x) else str(x)):
        sub = [r for r in rows if r[group_key] == G]
        o, t = _summarise_one(sub, reference, metrics, tests_on)
        for rec in o:
            rec[group_key] = G
        out += o
        tests[f"{group_key}_{G}"] = t
    return out, tests


def _isnum(x):
    try:
        float(x); return True
    except (TypeError, ValueError):
        return False


def _summarise_one(rows, reference, metrics, tests_on):
    by_method = defaultdict(list)
    for r in rows:
        by_method[r["method"]].append(r)
    out, tests = [], {}
    ref_rows = {r["seed"]: r for r in by_method.get(reference, [])}
    for m, rs in by_method.items():
        rec = {"method": m, "kind": rs[0]["kind"], "n_seeds": len(rs), "n_events": rs[0]["n_events"],
               "uav_hours": rs[0]["uav_hours"]}
        for k in metrics:
            v = np.array([float(r[k]) if r.get(k) is not None else np.nan for r in rs], dtype=float)
            rec[f"{k}_mean"] = float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")
            rec[f"{k}_std"] = float(np.nanstd(v, ddof=1)) if np.isfinite(v).sum() > 1 else 0.0
        out.append(rec)
    for met in tests_on:
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


def select_tasks(results_dir, methods=None, seeds=None):
    """Finished tasks to evaluate.  Default (methods=None): every main/ablation task; recorded variants
    (METHODS group "variant", e.g. TR-GAT-SC) are left out so they neither enter the Bonferroni family nor the
    per-method macros unless named explicitly."""
    return [(m, s, d) for m, s, d in list_tasks(results_dir)
            if (m in methods if methods is not None else METHODS[m].group != "variant")
            and (seeds is None or s in seeds)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--lead_s", type=float, default=10.0, help="lead time (s) that counts as timely")
    ap.add_argument("--persistence", nargs="+", type=int, default=[1, 3],
                    help="M-of-M alert persistence filters to report at the raw operating point")
    ap.add_argument("--soc_methods", nargs="+", default=SOC_METHODS,
                    help="methods for which the SOC curve / budget selection is computed (needs the val split)")
    ap.add_argument("--alphas", nargs="+", type=float, default=[1.0, 0.5, 0.25], help="EMA factors (1 = raw)")
    ap.add_argument("--hystereses", nargs="+", type=float, default=[0.0, 0.15],
                    help="hysteresis widths (switch-off threshold = threshold - h)")
    ap.add_argument("--grid", type=int, default=21, help="thresholds per SOC curve (quantiles of positive scores)")
    ap.add_argument("--budgets", nargs="+", type=float, default=[20.0, 50.0, 100.0],
                    help="false alert episodes per UAV-hour allowed on the validation split")
    ap.add_argument("--match", default="CPA-Rule",
                    help="rule whose validation false-episode rate is used as an extra (matched) budget")
    ap.add_argument("--reference", default="TR-GAT")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _smoke)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = select_tasks(args.results_dir, args.methods, args.seeds)
    if not tasks:
        raise SystemExit(f"no finished tasks under {args.results_dir}")
    # the matched rule first, so its validation false-episode rate is known before the learned methods
    tasks.sort(key=lambda t: (t[0] != args.match, t[0], t[1]))
    print(f"{len(tasks)} task(s) on {device}; lead_s={args.lead_s}; budgets={args.budgets} (+match {args.match})")

    t_start = time.perf_counter()
    rows, cause_rows, soc_rows, budget_rows, leads, data_cache = [], [], [], [], {}, {}
    match_budget = {}            # seed -> validation false-episode rate of the matched rule
    for method, seed, d in tasks:
        t0 = time.perf_counter()
        lm = load_task(d, device)
        cfg = lm.cfg
        splits = {}
        for split in ("val", "test"):
            key = cache_key(cfg, split)
            if key not in data_cache:
                data_cache[key] = get_split(cfg, split, cache_dir=args.cache_dir, verbose=False)
            splits[split] = data_cache[key]
        if lm.kind == "rule" and hasattr(lm.model, "fit") and "rule_config" not in (lm.metrics or {}).get("training", {}):
            lm.model.fit(splits["val"])
        n_scen = cfg.data.split_scenarios("test")
        if len(splits["test"]) % n_scen != 0:
            raise SystemExit(f"{len(splits['test'])} snapshots not divisible into {n_scen} scenarios")
        sps = len(splits["test"]) // n_scen
        dt = float(cfg.data.scenario_duration_s) / sps
        common = {"method": method, "kind": lm.kind, "seed": seed, "checkpoint_epoch": lm.checkpoint_epoch,
                  "snapshots": len(splits["test"]), "scenarios": n_scen, "snapshot_dt_s": dt, "lead_s": args.lead_s}

        # ---- (1) raw operating point on test
        test_tab = score_table(lm, splits["test"], sps)
        for M in args.persistence:
            res = event_metrics(test_tab, lead_s=args.lead_s, snapshot_dt_s=dt, persistence=M)
            rows.append({**common, **res.to_row()})
            for cname, cv in res.per_cause.items():
                cause_rows.append({"method": method, "seed": seed, "persistence": M, "cause": cname, **cv})
            leads[f"{method}/seed{seed}/p{M}"] = res.lead_s_list
            print(f"{method:12s} seed={seed} M={M}: events={res.n_events} event-CDR={res.event_cdr:.3f} "
                  f"timely={res.timely_cdr:.3f} lead med={res.lead_s['median']:.1f}s | episodes={res.n_alert_episodes} "
                  f"prec={res.episode_precision:.3f} false/UAV-h={res.false_episodes_per_uav_hour:.2f}")

        if method not in args.soc_methods:
            print(f"   ({time.perf_counter() - t0:.0f}s)")
            continue

        # ---- (2) SOC curves on val (for selection) and test (for the figure)
        n_val_scen = cfg.data.split_scenarios("val")
        val_tab = score_table(lm, splits["val"], len(splits["val"]) // n_val_scen)
        val_dt = float(cfg.data.scenario_duration_s) / (len(splits["val"]) // n_val_scen)
        val_points = []
        if lm.kind == "rule":
            configs = [(1.0, 0.0)]                      # rules output {0,1}: a single operating point
        else:
            configs = [(a, h) for a in args.alphas for h in args.hystereses]
        for alpha, hyst in configs:
            grid = threshold_grid(val_tab, args.grid) if lm.kind != "rule" else np.array([0.5])
            vp = soc_curve(val_tab, grid, alpha, hyst, lead_s=args.lead_s, snapshot_dt_s=val_dt)
            val_points += vp
            tp = soc_curve(test_tab, grid, alpha, hyst, lead_s=args.lead_s, snapshot_dt_s=dt)
            for p in tp:
                soc_rows.append({**common, "split": "test", **p.to_row()})
            for p in vp:
                soc_rows.append({**common, "split": "val", **p.to_row()})
        raw_val = event_metrics(val_tab, lead_s=args.lead_s, snapshot_dt_s=val_dt)
        if method == args.match:
            match_budget[seed] = raw_val.false_episodes_per_uav_hour
            print(f"   matched budget (val false episodes/UAV-h of {method}, seed {seed}): {match_budget[seed]:.2f}")

        # ---- (3) budget-matched operating points (selected on val, evaluated on test)
        budgets = [(f"fixed_{b:g}", b) for b in args.budgets]
        mb = match_budget.get(seed, match_budget.get(min(match_budget), None) if match_budget else None)
        if mb is not None:
            budgets.append((f"match_{args.match}", mb))
        for kind_b, b in budgets:
            sel = select_operating_point(val_points, b)
            if sel is None:
                budget_rows.append({**common, "budget_kind": kind_b, "budget": b, "feasible": 0})
                continue
            a = filtered_alerts(test_tab, sel.threshold, sel.alpha, sel.hysteresis)
            res = event_metrics(test_tab, lead_s=args.lead_s, snapshot_dt_s=dt, alert=a,
                                threshold=sel.threshold, alpha=sel.alpha, hysteresis=sel.hysteresis)
            budget_rows.append({**common, "budget_kind": kind_b, "budget": b, "feasible": 1,
                                "val_event_cdr": sel.event_cdr, "val_false_episodes_per_uav_hour":
                                sel.false_episodes_per_uav_hour, **res.to_row()})
            leads[f"{method}/seed{seed}/budget{b:g}"] = res.lead_s_list
            print(f"   budget {kind_b}={b:.2f}: alpha={sel.alpha:g} h={sel.hysteresis:g} thr={sel.threshold:.3f} -> "
                  f"test event-CDR={res.event_cdr:.3f} timely={res.timely_cdr:.3f} lead med={res.lead_s['median']:.1f}s "
                  f"false/UAV-h={res.false_episodes_per_uav_hour:.2f}")
        print(f"   ({time.perf_counter() - t0:.0f}s)")

    O = Path(args.out_dir)
    write_csv(O / f"events{args.tag}.csv", rows)
    write_csv(O / f"events_by_cause{args.tag}.csv", cause_rows)
    summary, tests = summarise(rows, args.reference, "persistence")
    write_csv(O / f"events_summary{args.tag}.csv", summary,
              sorted({k for r in summary for k in r}, key=lambda k: (k not in ("method", "kind", "persistence", "n_seeds"), k)))
    np.savez_compressed(O / f"events_lead{args.tag}.npz", **{k: v for k, v in leads.items() if v is not None})
    write_csv(O / f"events_soc{args.tag}.csv", soc_rows)
    b_tests = {}
    if budget_rows:
        write_csv(O / f"events_budget{args.tag}.csv", budget_rows,
                  sorted({k for r in budget_rows for k in r}, key=lambda k: (k not in ("method", "kind", "seed", "budget_kind", "budget"), k)))
        feasible = [r for r in budget_rows if r.get("feasible")]
        if feasible:
            b_summary, b_tests = summarise(feasible, args.reference, "budget_kind",
                                           metrics=SUMMARY_METRICS + ("threshold", "alpha", "hysteresis", "budget"))
            write_csv(O / f"events_budget_summary{args.tag}.csv", b_summary,
                      sorted({k for r in b_summary for k in r}, key=lambda k: (k not in ("method", "kind", "budget_kind", "n_seeds"), k)))
    meta = {"results_dir": args.results_dir, "lead_s": args.lead_s, "reference": args.reference,
            "persistence": args.persistence, "alphas": args.alphas, "hystereses": args.hystereses, "grid": args.grid,
            "budgets": args.budgets, "match": args.match, "match_budget_val": match_budget,
            "soc_methods": args.soc_methods, "tasks": [str(d) for _, _, d in tasks],
            "tests": tests, "budget_tests": b_tests, "env": env_info(device),
            "seconds": time.perf_counter() - t_start, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(O / f"events{args.tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"written: {O}/events{args.tag}.csv, events_by_cause, events_summary, events_lead.npz, events_soc, "
          f"events_budget(_summary), events.json ({time.perf_counter() - t_start:.0f}s)")


if __name__ == "__main__":
    main()
