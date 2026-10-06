#!/usr/bin/env python3
"""Aggregate the main experiment (S7b).

Reads  results/<results_dir>/<method>/seed<n>/metrics.json  (+ optional
results/eval/<method>/seed<n>.json for latency measured by eval_only.py) and writes

  <out_prefix>_summary.csv     mean ± std over seeds, bootstrap 95% CI, paired t-tests
  <out_prefix>_per_regime.csv  per-regime / per-cause CDR
  <out_prefix>_tests.json      full test statistics, host check, provenance

Latency figures are only pooled if every run comes from the same GPU model
and host; otherwise the script raises (use --allow_mixed_hosts to drop the
latency columns instead).
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from skyflow.experiments.loader import list_tasks
from skyflow.experiments.methods import MAIN_METHODS, METHODS
from skyflow.experiments.stats import bonferroni, bootstrap_ci, group_snapshots, paired_ttest

SCALARS = ("cdr", "far", "f1", "precision", "num_missed_positives", "auprc", "threshold")


def load_runs(results_dir, eval_dir, methods):
    runs = defaultdict(dict)
    for m, s, d in list_tasks(results_dir):
        if m not in methods:
            continue
        rec = json.load(open(d / "metrics.json", encoding="utf-8"))
        ev = Path(eval_dir) / m / f"seed{s}.json" if eval_dir else None
        rec["_eval"] = json.load(open(ev, encoding="utf-8")) if ev and ev.exists() else None
        runs[m][s] = rec
    return runs


def host_check(runs, allow_mixed):
    hosts = set()
    for m, by_seed in runs.items():
        for s, r in by_seed.items():
            src = r["_eval"]["env"] if r["_eval"] else r["env"]
            hosts.add((src.get("hostname"), src.get("gpu_model")))
    hosts_sorted = sorted(hosts, key=str)      # entries may contain None (e.g. CPU-only rule runs)
    if len(hosts) > 1:
        msg = f"latency data come from different machines/GPUs: {hosts_sorted}"
        if not allow_mixed:
            raise SystemExit("ERROR: " + msg + "  (re-run eval_only.py on one machine, or --allow_mixed_hosts)")
        print("WARNING: " + msg + " -> latency columns dropped")
        return None, hosts_sorted
    return next(iter(hosts)) if hosts else (None, None), hosts_sorted


def latency_of(r):
    """(p95_total, stage dict) from eval json if present else from training-time test eval."""
    if r["_eval"]:
        L = r["_eval"]["latency"]
        return L["p95_sum_ms"], {
            "graph_build": L["graph_build"]["graph_build_p95_ms"],
            "gnn_forward": L["inference"]["gnn_forward_p95_ms"],
            "pair_scoring": L["inference"]["pair_scoring_p95_ms"],
        }, "eval_only"
    t = r["test"]
    st = t.get("stage_p95_ms", {})
    return sum(v for v in st.values() if v is not None), st, "run_task"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--eval_dir", default="results/eval")
    ap.add_argument("--out_prefix", default="results/main")
    ap.add_argument("--methods", nargs="+", default=MAIN_METHODS)
    ap.add_argument("--reference", default="TR-GAT")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--allow_mixed_hosts", action="store_true")
    args = ap.parse_args()

    runs = load_runs(args.results_dir, args.eval_dir, args.methods)
    if not runs:
        raise SystemExit(f"no finished runs in {args.results_dir}")
    host, all_hosts = host_check(runs, args.allow_mixed_hosts)
    use_latency = host is not None

    # ---- per-method aggregation ------------------------------------------------
    rows, per_regime_rows, tests = [], [], {"reference": args.reference, "hosts": all_hosts}
    per_seed_metric = defaultdict(dict)   # method -> metric -> {seed: value}
    boot = {}
    for m in args.methods:
        if m not in runs:
            print(f"[warn] no runs for {m}")
            continue
        by_seed = runs[m]
        seeds = sorted(by_seed)
        vals = {k: np.array([by_seed[s]["test"].get(k, np.nan) for s in seeds], dtype=float) for k in SCALARS}
        for k in SCALARS:
            per_seed_metric[m][k] = {s: by_seed[s]["test"].get(k, np.nan) for s in seeds}
        row = {"method": m, "kind": METHODS[m].kind, "n_seeds": len(seeds), "seeds": " ".join(map(str, seeds))}
        for k in SCALARS:
            row[f"{k}_mean"] = vals[k].mean()
            row[f"{k}_std"] = vals[k].std(ddof=1) if len(seeds) > 1 else 0.0
        # bootstrap over test scenarios
        try:
            n_scen = by_seed[seeds[0]]["dataset"]["test_scenarios"]
            counts = [group_snapshots(by_seed[s]["test"]["per_snapshot"], n_scen) for s in seeds]
            ci = bootstrap_ci(counts, n_boot=args.n_boot)
            boot[m] = ci
            for k in ("cdr", "far", "f1"):
                row[f"{k}_ci_lo"], row[f"{k}_ci_hi"] = ci[k]["lo"], ci[k]["hi"]
        except (KeyError, ValueError) as e:
            print(f"[warn] bootstrap skipped for {m}: {e}")
        # latency
        if use_latency:
            lat = [latency_of(by_seed[s]) for s in seeds]
            row["latency_p95_ms"] = np.mean([l[0] for l in lat])
            for st in ("graph_build", "gnn_forward", "pair_scoring"):
                row[f"{st}_p95_ms"] = np.mean([l[1].get(st) or 0.0 for l in lat])
            row["latency_source"] = lat[0][2]
        # training info
        tr = [by_seed[s]["training"] for s in seeds]
        row["params"] = by_seed[seeds[0]]["num_parameters"]
        row["train_min_mean"] = np.mean([t.get("train_seconds", 0.0) for t in tr]) / 60
        row["epochs_run_mean"] = np.mean([t.get("epochs_run", 0) for t in tr])
        row["best_epoch_mean"] = np.mean([t.get("best_epoch", 0) for t in tr])
        row["git_commits"] = " ".join(sorted({by_seed[s]["env"]["git_commit"][:8] for s in seeds}))
        rows.append(row)
        # per regime
        regimes = sorted({r for s in seeds for r in by_seed[s]["test"]["per_regime"]})
        for reg in regimes:
            cd = np.array([by_seed[s]["test"]["per_regime"].get(reg, {}).get("cdr", np.nan) for s in seeds])
            npos = np.array([by_seed[s]["test"]["per_regime"].get(reg, {}).get("num_positives", 0) for s in seeds])
            per_regime_rows.append({"method": m, "regime": reg, "cdr_mean": np.nanmean(cd),
                                    "cdr_std": np.nanstd(cd, ddof=1) if len(seeds) > 1 else 0.0,
                                    "num_positives": npos.mean(), "n_seeds": len(seeds)})

    # ---- paired t-tests vs reference ------------------------------------------
    ref = args.reference
    if ref in per_seed_metric:
        for k in ("f1", "cdr", "far"):
            raw = {}
            for m in per_seed_metric:
                if m == ref:
                    continue
                common = sorted(set(per_seed_metric[ref][k]) & set(per_seed_metric[m][k]))
                if len(common) < 2:
                    raw[m] = {"n": len(common), "p": float("nan"), "t": float("nan"), "mean_diff": float("nan")}
                    continue
                raw[m] = paired_ttest([per_seed_metric[ref][k][s] for s in common],
                                      [per_seed_metric[m][k][s] for s in common])
            adj = bonferroni({m: v["p"] for m, v in raw.items()})
            for m in raw:
                raw[m]["p_bonferroni"] = adj[m]
            tests[f"paired_t_{k}"] = raw
            for row in rows:
                if row["method"] in raw:
                    row[f"p_{k}_vs_ref"] = raw[row["method"]]["p_bonferroni"]
    tests["bootstrap"] = boot
    tests["n_boot"] = args.n_boot

    # ---- write -------------------------------------------------------------------
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = []
    for r in rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    with open(f"{out}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in r.items()})
    with open(f"{out}_per_regime.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["method", "regime", "cdr_mean", "cdr_std", "num_positives", "n_seeds"])
        w.writeheader()
        for r in per_regime_rows:
            w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in r.items()})
    with open(f"{out}_tests.json", "w", encoding="utf-8") as f:
        json.dump(tests, f, indent=2, default=float)

    print(f"{'method':20s} {'n':>2s} {'CDR':>14s} {'FAR':>14s} {'F1':>14s} {'P95 ms':>8s} {'p(F1)':>8s}")
    for r in rows:
        lat = f"{r['latency_p95_ms']:8.1f}" if "latency_p95_ms" in r else "     n/a"
        p = f"{r['p_f1_vs_ref']:8.3g}" if "p_f1_vs_ref" in r else "     ref"
        print(f"{r['method']:20s} {r['n_seeds']:2d} {r['cdr_mean']:.4f}+-{r['cdr_std']:.4f} "
              f"{r['far_mean']:.4f}+-{r['far_std']:.4f} {r['f1_mean']:.4f}+-{r['f1_std']:.4f} {lat} {p}")
    print(f"written: {out}_summary.csv, {out}_per_regime.csv, {out}_tests.json")


if __name__ == "__main__":
    main()
