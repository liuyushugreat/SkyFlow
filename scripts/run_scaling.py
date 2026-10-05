#!/usr/bin/env python3
"""Latency scaling with fleet size N (S7b).

For each N: time graph construction (builder + proximity candidates, CPU) and
TR-GAT inference (gnn_forward + pair_scoring, --device) over --epochs
observation epochs, record mean degree, edge count and candidate-pair count,
then fit log-log slopes alpha (y = c * N^alpha) per stage and for the total.

Out-of-memory at large N is recorded as a row with status=oom and the error
message; it is never silently skipped.

  --mode fixed_area        area fixed at config grid_size_m (density grows with N)   [default]
  --mode constant_density  area scaled so that N / area stays at the config value

    python scripts/run_scaling.py --checkpoint results/main/TR-GAT/seed42 --sizes 100 250 500 1000 2000
"""

import argparse
import csv
import json
import math
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from skyflow.experiments.env_info import env_info
from skyflow.experiments.latency import build_latency_dataset, measure_graph_build, measure_inference
from skyflow.experiments.loader import load_task
from skyflow.experiments.stats import fit_power_law

FIELDS = ["num_uavs", "grid_size_m", "mode", "status", "graph_build_p95_ms", "builder_only_p95_ms",
          "candidates_only_p95_ms", "gnn_forward_p95_ms", "pair_scoring_p95_ms", "total_p95_ms",
          "graph_build_mean_ms", "gnn_forward_mean_ms", "pair_scoring_mean_ms", "total_mean_ms",
          "mean_degree", "mean_edges", "mean_candidate_pairs", "epochs_build", "epochs_infer",
          "distinct_snapshots", "peak_gpu_mem_gb", "seconds", "error"]


def _is_oom(e: BaseException) -> bool:
    return isinstance(e, (MemoryError, torch.cuda.OutOfMemoryError)) or "out of memory" in str(e).lower()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="task dir, e.g. results/main/TR-GAT/seed42")
    ap.add_argument("--sizes", nargs="+", type=int, default=[100, 250, 500, 1000, 2000])
    ap.add_argument("--epochs", type=int, default=1000, help="observation epochs per N for P95")
    ap.add_argument("--distinct", type=int, default=100, help="distinct snapshots for inference timing")
    ap.add_argument("--mode", choices=["fixed_area", "constant_density"], default="fixed_area")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out_csv", default="results/scaling.csv")
    ap.add_argument("--out_fit", default="results/scaling_fit.json")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    lm = load_task(args.checkpoint, device)
    base_cfg = lm.cfg
    base_n, base_grid = base_cfg.data.num_uavs, base_cfg.data.grid_size_m
    rows = []
    for n in args.sizes:
        import copy
        cfg = copy.deepcopy(base_cfg)
        cfg.data.num_uavs = n
        if args.mode == "constant_density":
            cfg.data.grid_size_m = float(base_grid * math.sqrt(n / base_n))
        row = {k: "" for k in FIELDS}
        row.update({"num_uavs": n, "grid_size_m": cfg.data.grid_size_m, "mode": args.mode, "status": "ok",
                    "epochs_build": 0, "epochs_infer": 0, "distinct_snapshots": 0, "error": ""})
        t0 = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        try:
            print(f"[N={n}] graph construction over {args.epochs} epochs (grid {cfg.data.grid_size_m:.0f} m) ...")
            gb = measure_graph_build(cfg, args.epochs, device=torch.device("cpu"), num_uavs=n)
            row.update({"graph_build_p95_ms": gb["graph_build_p95_ms"], "builder_only_p95_ms": gb["builder_only_p95_ms"],
                        "candidates_only_p95_ms": gb["candidates_only_p95_ms"], "graph_build_mean_ms": gb["graph_build_mean_ms"],
                        "mean_degree": gb["mean_degree"], "mean_edges": gb["mean_edges"],
                        "mean_candidate_pairs": gb["mean_candidate_pairs"], "epochs_build": gb["n_epochs"]})
            print(f"[N={n}] inference over {args.epochs} epochs ({args.distinct} distinct snapshots) ...")
            data = build_latency_dataset(cfg, num_uavs=n, n_distinct=args.distinct)
            inf = measure_inference(lm.evaluate, data, args.epochs, device=device)
            row.update({"gnn_forward_p95_ms": inf["gnn_forward_p95_ms"], "pair_scoring_p95_ms": inf["pair_scoring_p95_ms"],
                        "gnn_forward_mean_ms": inf["gnn_forward_mean_ms"], "pair_scoring_mean_ms": inf["pair_scoring_mean_ms"],
                        "epochs_infer": inf["n_epochs"], "distinct_snapshots": len(data)})
            row["total_p95_ms"] = gb["graph_build_p95_ms"] + inf["gnn_forward_p95_ms"] + inf["pair_scoring_p95_ms"]
            row["total_mean_ms"] = gb["graph_build_mean_ms"] + inf["gnn_forward_mean_ms"] + inf["pair_scoring_mean_ms"]
            del data
        except BaseException as e:  # noqa: BLE001 - we must record, not hide
            if isinstance(e, KeyboardInterrupt):
                raise
            row["status"] = "oom" if _is_oom(e) else "error"
            row["error"] = f"{type(e).__name__}: {e}"
            print(f"[N={n}] FAILED ({row['status']}): {row['error']}")
            traceback.print_exc()
        if device.type == "cuda":
            row["peak_gpu_mem_gb"] = torch.cuda.max_memory_allocated(device) / 1e9
        row["seconds"] = time.perf_counter() - t0
        rows.append(row)
        if row["status"] == "ok":
            print(f"[N={n}] P95 build={row['graph_build_p95_ms']:.1f} fwd={row['gnn_forward_p95_ms']:.1f} "
                  f"score={row['pair_scoring_p95_ms']:.1f} total={row['total_p95_ms']:.1f} ms | "
                  f"k={row['mean_degree']:.1f} cand={row['mean_candidate_pairs']:.0f} ({row['seconds']:.0f}s)")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    ok = [r for r in rows if r["status"] == "ok"]
    ns = [r["num_uavs"] for r in ok]
    fits = {}
    for key in ("total_p95_ms", "graph_build_p95_ms", "gnn_forward_p95_ms", "pair_scoring_p95_ms",
                "total_mean_ms", "mean_edges", "mean_candidate_pairs"):
        fits[key] = fit_power_law(ns, [r[key] for r in ok])
    fit = {"mode": args.mode, "sizes_requested": args.sizes, "sizes_ok": ns,
           "failed": [{"num_uavs": r["num_uavs"], "status": r["status"], "error": r["error"]} for r in rows if r["status"] != "ok"],
           "epochs": args.epochs, "distinct_snapshots": args.distinct, "checkpoint": str(lm.task_dir),
           "fits": fits, "env": env_info(device), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(args.out_fit, "w", encoding="utf-8") as f:
        json.dump(fit, f, indent=2, default=str)
    print("alpha (log-log slope):", {k: round(v["alpha"], 3) for k, v in fits.items() if v["alpha"] == v["alpha"]})
    print(f"written: {out_csv}, {args.out_fit}")


if __name__ == "__main__":
    main()
