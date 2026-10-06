#!/usr/bin/env python3
"""Latency / packet-loss robustness sweeps (S7b).  Evaluation only: models
trained under the nominal condition are tested on the SAME truth scenarios
re-observed under degraded ADS-B conditions (labels are unchanged because
they depend only on the truth).

  latency sweep : mean ADS-B latency L in {0, 0.5, 1, 2, 3} s, per-UAV latency
                  ~ U(L(1-j), L(1+j)) with jitter j (--latency_jitter, 0.4)
  loss sweep    : per-report loss probability in {0, 0.1, 0.2, 0.3} at nominal latency

Outputs results/robustness_latency.csv and results/robustness_loss.csv
(one row per condition x method x seed) plus a .json with provenance.

    python scripts/run_robustness.py --results_dir results/main
"""

import argparse
import copy
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from skyflow.data.cache import cache_key, get_split
from skyflow.experiments.env_info import env_info
from skyflow.experiments.loader import list_tasks, load_task

DEFAULT_METHODS = ["TR-GAT", "TR-GAT-NT", "GAT-S", "CPA-Rule", "Plan-CPA"]
FIELDS = ["sweep", "value", "adsb_latency_lo_s", "adsb_latency_hi_s", "packet_loss", "method", "kind", "seed",
          "cdr", "far", "f1", "precision", "num_pairs", "num_positives", "num_missed_positives",
          "test_snapshots", "cache_key"]


def condition_cfg(base, sweep, value, jitter):
    cfg = copy.deepcopy(base)
    if sweep == "latency":
        lo, hi = max(value * (1 - jitter), 0.0), value * (1 + jitter)
        cfg.sim.adsb_latency_s = [float(lo), float(hi)]
    elif sweep == "loss":
        cfg.sim.packet_loss = float(value)
    else:
        raise ValueError(sweep)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--out_dir", default="results")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--latencies", nargs="+", type=float, default=[0.0, 0.5, 1.0, 2.0, 3.0])
    ap.add_argument("--losses", nargs="+", type=float, default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--latency_jitter", type=float, default=0.4)
    ap.add_argument("--sweeps", nargs="+", default=["latency", "loss"], choices=["latency", "loss"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for output files (e.g. _smoke)")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = [(m, s, d) for m, s, d in list_tasks(args.results_dir)
             if m in args.methods and (args.seeds is None or s in args.seeds)]
    if not tasks:
        raise SystemExit(f"no finished tasks for {args.methods} under {args.results_dir}")
    loaded = []
    for m, s, d in tasks:
        lm = load_task(d, device)
        if lm.kind == "rule" and hasattr(lm.model, "fit") and "rule_config" not in (lm.metrics or {}).get("training", {}):
            lm.model.fit(get_split(lm.cfg, "val", cache_dir=args.cache_dir, verbose=False))
        loaded.append(lm)
    print(f"{len(loaded)} checkpoint(s): " + ", ".join(f"{l.method}/s{l.seed}" for l in loaded))

    t_start = time.perf_counter()
    for sweep in args.sweeps:
        values = args.latencies if sweep == "latency" else args.losses
        rows = []
        for v in values:
            datasets = {}
            for lm in loaded:
                cfg = condition_cfg(lm.cfg, sweep, v, args.latency_jitter)
                k = cache_key(cfg, "test")
                if k not in datasets:
                    print(f"[{sweep}={v}] building/loading test split (key {k}) ...")
                    datasets[k] = get_split(cfg, "test", cache_dir=args.cache_dir, verbose=False)
                test = datasets[k]
                res = lm.evaluate(test)
                lat = cfg.sim.adsb_latency_s
                lo, hi = (lat if isinstance(lat, (list, tuple)) else (lat, lat))
                rows.append({
                    "sweep": sweep, "value": v, "adsb_latency_lo_s": lo, "adsb_latency_hi_s": hi,
                    "packet_loss": cfg.sim.packet_loss, "method": lm.method, "kind": lm.kind, "seed": lm.seed,
                    "cdr": res.cdr, "far": res.far, "f1": res.f1, "precision": res.precision,
                    "num_pairs": res.num_pairs, "num_positives": res.num_positives,
                    "num_missed_positives": res.num_missed_positives, "test_snapshots": len(test), "cache_key": k,
                })
                print(f"  {sweep}={v:<4} {lm.method:10s} seed={lm.seed}: CDR={res.cdr:.4f} FAR={res.far:.4f} F1={res.f1:.4f}")
        out = Path(args.out_dir) / f"robustness_{sweep}{args.tag}.csv"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        meta = {"sweep": sweep, "values": values, "latency_jitter": args.latency_jitter, "methods": args.methods,
                "results_dir": args.results_dir, "checkpoints": [str(l.task_dir) for l in loaded],
                "env": env_info(device), "seconds": time.perf_counter() - t_start,
                "created": time.strftime("%Y-%m-%d %H:%M:%S")}
        with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        print(f"written: {out}")


if __name__ == "__main__":
    main()
