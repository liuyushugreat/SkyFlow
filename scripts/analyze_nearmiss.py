#!/usr/bin/env python3
"""Near-miss analysis of false alerts (S8f): how close did the "false" pairs
actually come?  Re-integrates the deterministic test truth and reports the
normalised separation rho (1 = separation standard) of false alert rows and
episodes against a random negative sample.  Labels are untouched; a
consistency check confirms the re-simulated truth reproduces them.

    python scripts/analyze_nearmiss.py --checkpoints results/main/TR-GAT/seed42 results/main/CPA-Rule/seed42 \
        --out_csv results/nearmiss.csv
Optional: --budget_json results/events_budget.json to also analyse the
budget-matched operating point of each checkpoint.
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from skyflow.data.cache import get_split
from skyflow.experiments.env_info import env_info
from skyflow.experiments.events import filtered_alerts, score_table
from skyflow.experiments.loader import load_task
from skyflow.experiments.nearmiss import RHO_LEVELS, near_miss_analysis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out_csv", default="results/nearmiss.csv")
    ap.add_argument("--budget_csv", default=None,
                    help="events_budget.csv: also analyse the budget-matched operating point of each checkpoint")
    ap.add_argument("--n_sample", type=int, default=200_000)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    budget_ops = {}
    if args.budget_csv and Path(args.budget_csv).exists():
        import pandas as pd
        b = pd.read_csv(args.budget_csv)
        for _, r in b.iterrows():
            budget_ops.setdefault((r["method"], int(r["seed"])), []).append(r.to_dict())

    rows, meta_cp, logs_cache = [], [], {}
    t0 = time.perf_counter()
    for cp in args.checkpoints:
        lm = load_task(cp, device)
        cfg = lm.cfg
        if lm.kind == "rule" and hasattr(lm.model, "fit") and "rule_config" not in (lm.metrics or {}).get("training", {}):
            lm.model.fit(get_split(cfg, "val", cache_dir=args.cache_dir, verbose=False))
        data = get_split(cfg, args.split, cache_dir=args.cache_dir, verbose=False)
        n_scen = cfg.data.split_scenarios(args.split)
        sps = len(data) // n_scen
        epoch_step = int(round(cfg.data.scenario_duration_s * cfg.data.sim_freq_hz / sps))
        horizon = int(round(cfg.data.lookahead_seconds * cfg.data.sim_freq_hz))
        sim_key = (cfg.data.num_uavs, cfg.data.grid_size_m, getattr(cfg.data, "sim_seed", cfg.training.seed), n_scen)
        if sim_key not in logs_cache:
            sim = cfg.make_simulator(seed=getattr(cfg.data, "sim_seed", cfg.training.seed))
            logs_cache[sim_key] = sim.simulate_logs(args.split, n_scen, cfg.data.scenario_duration_s)
        logs = logs_cache[sim_key]
        tab = score_table(lm, data, sps)
        ops = [("val_f1", tab.alert, {"threshold": tab.threshold, "alpha": 1.0, "hysteresis": 0.0})]
        for r in budget_ops.get((lm.method, lm.seed), []):
            # rules are single operating points: fixed budgets they cannot meet are recorded as infeasible (NaN)
            if int(r.get("feasible", 1)) == 0 or not all(math.isfinite(float(r[k])) for k in ("threshold", "alpha", "hysteresis")):
                continue
            a = filtered_alerts(tab, float(r["threshold"]), float(r["alpha"]), float(r["hysteresis"]))
            ops.append((f"budget_{float(r['budget']):g}", a, {k: r[k] for k in ("threshold", "alpha", "hysteresis")}))
        for op_name, alert, tags in ops:
            res = near_miss_analysis(tab, alert, logs, epoch_step, horizon, cfg.data.conflict_h_sep_m,
                                     cfg.data.conflict_v_sep_m, n_negative_sample=args.n_sample)
            for r in res.rows():
                rows.append({"method": lm.method, "seed": lm.seed, "operating_point": op_name, **tags, **r,
                             "positive_mismatch": res.positive_mismatch, "negative_mismatch": res.negative_mismatch,
                             "n_positive_checked": res.n_positive_checked, "n_negative_checked": res.n_negative_checked})
            print(f"{lm.method:12s} seed={lm.seed} [{op_name}] label check: pos mismatch {res.positive_mismatch}/"
                  f"{res.n_positive_checked}, neg mismatch {res.negative_mismatch}/{res.n_negative_checked}")
            for r in res.rows():
                print(f"   {r['group']:16s} n={r['n']:8d} median rho={r['rho_median']:.2f} "
                      + " ".join(f"<{lv:g}:{r[f'frac_lt_{lv:g}']:.3f}" for lv in RHO_LEVELS))
        meta_cp.append(str(lm.task_dir))

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    meta = {"checkpoints": meta_cp, "split": args.split, "levels": list(RHO_LEVELS), "n_sample": args.n_sample,
            "budget_csv": args.budget_csv, "env": env_info(device), "seconds": time.perf_counter() - t0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"written: {out}")


if __name__ == "__main__":
    main()
