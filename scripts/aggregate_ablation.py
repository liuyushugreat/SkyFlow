#!/usr/bin/env python3
"""Ablation table (S7b): full TR-GAT vs. each switched-off component.

Rows: TR-GAT (full), TR-GAT-NT (= no temporal encoding, reused from the main
runs), abl_no_gating, abl_no_gru, abl_bce, abl_telemetry_only.  Reports
mean +- std over seeds, the delta to the full model and a paired t-test
(Bonferroni over the ablations).  Output: results/ablation.csv (+ _tests.json).

    python scripts/aggregate_ablation.py --results_dir results/main
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from skyflow.experiments.loader import list_tasks
from skyflow.experiments.methods import ABLATION_METHODS, METHODS
from skyflow.experiments.stats import bonferroni, paired_ttest

LABELS = {"TR-GAT": "full", "TR-GAT-NT": "no temporal encoding", "abl_no_gating": "no relation gating",
          "abl_no_gru": "no GRU recurrence", "abl_bce": "BCE instead of focal", "abl_telemetry_only": "telemetry only",
          "abl_no_plan": "no filed-plan context"}
FIELDS = ["method", "variant", "n_seeds", "seeds", "cdr_mean", "cdr_std", "far_mean", "far_std", "f1_mean", "f1_std",
          "d_cdr", "d_far", "d_f1", "p_f1_bonf", "p_cdr_bonf", "p_far_bonf", "auprc_mean", "threshold_mean",
          "params", "epochs_run_mean", "git_commits"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--out_csv", default="results/ablation.csv")
    ap.add_argument("--reference", default="TR-GAT")
    ap.add_argument("--methods", nargs="+", default=["TR-GAT", "TR-GAT-NT"] + ABLATION_METHODS)
    args = ap.parse_args()

    runs = {}
    for m, s, d in list_tasks(args.results_dir):
        if m in args.methods:
            runs.setdefault(m, {})[s] = json.load(open(d / "metrics.json", encoding="utf-8"))
    if args.reference not in runs:
        raise SystemExit(f"reference {args.reference} has no finished runs in {args.results_dir}")
    ref = runs[args.reference]

    rows, raw_p = [], {"f1": {}, "cdr": {}, "far": {}}
    for m in args.methods:
        if m not in runs:
            print(f"[warn] no runs for {m}")
            continue
        by_seed = runs[m]
        seeds = sorted(by_seed)
        v = {k: np.array([by_seed[s]["test"][k] for s in seeds]) for k in ("cdr", "far", "f1")}
        r = {k: np.array([ref[s]["test"][k] for s in sorted(ref)]) for k in ("cdr", "far", "f1")}
        row = {"method": m, "variant": LABELS.get(m, METHODS[m].description), "n_seeds": len(seeds),
               "seeds": " ".join(map(str, seeds)), "params": by_seed[seeds[0]]["num_parameters"],
               "epochs_run_mean": np.mean([by_seed[s]["training"].get("epochs_run", 0) for s in seeds]),
               "git_commits": " ".join(sorted({by_seed[s]["env"]["git_commit"][:8] for s in seeds})),
               "auprc_mean": np.mean([by_seed[s]["test"].get("auprc", np.nan) for s in seeds]),
               "threshold_mean": np.mean([by_seed[s]["test"].get("threshold", np.nan) for s in seeds])}
        for k in ("cdr", "far", "f1"):
            row[f"{k}_mean"] = v[k].mean()
            row[f"{k}_std"] = v[k].std(ddof=1) if len(seeds) > 1 else 0.0
            row[f"d_{k}"] = v[k].mean() - r[k].mean()
            if m != args.reference:
                common = sorted(set(seeds) & set(ref))
                if len(common) >= 2:
                    raw_p[k][m] = paired_ttest([ref[s]["test"][k] for s in common], [by_seed[s]["test"][k] for s in common])
                else:
                    raw_p[k][m] = {"n": len(common), "p": float("nan"), "t": float("nan"), "mean_diff": float("nan")}
        rows.append(row)
    for k in ("f1", "cdr", "far"):
        adj = bonferroni({m: t["p"] for m, t in raw_p[k].items()})
        for m, t in raw_p[k].items():
            t["p_bonferroni"] = adj[m]
        for row in rows:
            row[f"p_{k}_bonf"] = adj.get(row["method"], "")

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({c: (f"{v:.6g}" if isinstance(v, float) else v) for c, v in row.items()})
    with open(out.with_name(out.stem + "_tests.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": args.reference, "paired_t": raw_p}, f, indent=2, default=float)
    print(f"{'variant':24s} {'n':>2s} {'F1':>14s} {'dF1':>8s} {'CDR':>14s} {'FAR':>14s} {'p(F1)':>8s}")
    for row in rows:
        p = row["p_f1_bonf"]
        print(f"{row['variant']:24s} {row['n_seeds']:2d} {row['f1_mean']:.4f}+-{row['f1_std']:.4f} {row['d_f1']:+8.4f} "
              f"{row['cdr_mean']:.4f}+-{row['cdr_std']:.4f} {row['far_mean']:.4f}+-{row['far_std']:.4f} "
              f"{(f'{p:8.3g}' if isinstance(p, float) else '     ref')}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
