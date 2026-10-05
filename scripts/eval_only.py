#!/usr/bin/env python3
"""Evaluate finished checkpoints on the test split and measure three-stage
P95 latency on THIS machine (S7b).  No training.

    python scripts/eval_only.py --results_dir results/main --out_dir results/eval
    python scripts/eval_only.py --results_dir results/main --methods TR-GAT --seeds 42 --latency_epochs 1000

Output: results/eval/{method}/seed{n}.json with test metrics, latency
(graph_build re-measured here, gnn_forward + pair_scoring over
``--latency_epochs`` snapshots) and hardware provenance.
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from skyflow.data.cache import get_split
from skyflow.experiments.env_info import env_info
from skyflow.experiments.latency import measure_graph_build, measure_inference
from skyflow.experiments.loader import list_tasks, load_task


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--out_dir", default="results/eval")
    ap.add_argument("--methods", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--latency_epochs", type=int, default=1000)
    ap.add_argument("--skip_graph_build", action="store_true",
                    help="do not re-measure graph construction (uses cached build_time_ms)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    tasks = [(m, s, d) for m, s, d in list_tasks(args.results_dir)
             if (args.methods is None or m in args.methods) and (args.seeds is None or s in args.seeds)]
    if not tasks:
        raise SystemExit(f"no finished tasks under {args.results_dir}")
    print(f"{len(tasks)} task(s) on {device}")

    data_cache = {}
    graph_cache = {}
    for method, seed, d in tasks:
        t0 = time.perf_counter()
        lm = load_task(d, device)
        cfg = lm.cfg
        key = json.dumps({"n": cfg.data.num_uavs, "g": cfg.data.grid_size_m, "f": cfg.features.input_set}, sort_keys=True)
        val = None
        if (lm.kind == "rule" and hasattr(lm.model, "fit")
                and "rule_config" not in (lm.metrics or {}).get("training", {})):
            val = get_split(cfg, "val", cache_dir=args.cache_dir, verbose=False)
            lm.model.fit(val)
        test = get_split(cfg, "test", cache_dir=args.cache_dir, verbose=False)

        res = lm.evaluate(test)
        if args.skip_graph_build:
            gb = {"graph_build_p95_ms": res.stage_p95_ms.get("graph_build"),
                  "graph_build_mean_ms": res.stage_ms.get("graph_build"), "source": "cached_build_time_ms"}
        else:
            if key not in graph_cache:
                graph_cache[key] = measure_graph_build(cfg, args.latency_epochs, device=torch.device("cpu"))
                graph_cache[key].pop("build_ms", None)
                graph_cache[key].pop("candidates_ms", None)
                graph_cache[key]["source"] = "measured_here"
            gb = graph_cache[key]
        inf = measure_inference(lm.evaluate, test, args.latency_epochs, device=device)
        p95_total = (gb["graph_build_p95_ms"] or 0.0) + (inf["gnn_forward_p95_ms"] or 0.0) + (inf["pair_scoring_p95_ms"] or 0.0)

        out = {
            "method": method, "kind": lm.kind, "seed": seed, "task_dir": str(d),
            "checkpoint_epoch": lm.checkpoint_epoch,
            "test": res.to_dict(),
            "latency": {
                "epochs": args.latency_epochs,
                "graph_build": gb,
                "inference": inf,
                "p95_sum_ms": p95_total,          # sum of per-stage P95 (conservative)
                "note": "graph_build timed on CPU (numpy); gnn_forward/pair_scoring on --device",
            },
            "env": env_info(device),
            "eval_seconds": time.perf_counter() - t0,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        od = Path(args.out_dir) / method
        od.mkdir(parents=True, exist_ok=True)
        with open(od / f"seed{seed}.json", "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"{method:20s} seed={seed}: CDR={res.cdr:.4f} FAR={res.far:.4f} F1={res.f1:.4f} | "
              f"P95 build={gb['graph_build_p95_ms']:.1f} fwd={inf['gnn_forward_p95_ms']:.1f} "
              f"score={inf['pair_scoring_p95_ms']:.1f} ms ({out['eval_seconds']:.0f}s)")


if __name__ == "__main__":
    main()
