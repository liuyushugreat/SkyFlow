#!/usr/bin/env python3
"""Attribute ground-truth conflicts to their (synthetic) cause.

Runs the physics for a number of scenarios, labels every snapshot with the
configured label definition and counts conflicts per cause.  No graph is
built, so this is cheap.  Writes

    results/conflict_sources.csv      (cause, count, share, ...)
    results/conflict_sources.json     (same + provenance: config, seed, git commit)

Usage:
    python scripts/conflict_source_stats.py --scenarios 5 --duration 60
    python scripts/conflict_source_stats.py --num_uavs 200 --split val
"""

import argparse
import csv
import json
import platform
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from skyflow.config import SkyFlowConfig
from skyflow.data.urbanair500 import CAUSES


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--num_uavs", type=int, default=None)
    ap.add_argument("--scenarios", type=int, default=5)
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--split", default="test")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epoch_step", type=int, default=10)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)
    sim = cfg.make_simulator(num_uavs=args.num_uavs, seed=args.seed)
    logs = sim.simulate_logs(args.split, args.scenarios, args.duration)

    counts = Counter()
    uav_counts = Counter()
    ttc_by_cause = {c: [] for c in CAUSES}
    n_snapshots = 0
    for log in logs:
        for code in log.causes.tolist():
            uav_counts[CAUSES[code]] += 1
        for epoch in range(0, log.n_epochs, args.epoch_step):
            n_snapshots += 1
            for ev in sim.label(log, epoch):
                counts[ev.cause] += 1
                ttc_by_cause[ev.cause].append(ev.time_to_conflict)

    total = sum(counts.values())
    rows = []
    for c in CAUSES:
        n = counts.get(c, 0)
        ttc = np.array(ttc_by_cause[c]) if ttc_by_cause[c] else np.zeros(0)
        rows.append({
            "cause": c,
            "count": n,
            "share": n / total if total else 0.0,
            "uav_share": uav_counts.get(c, 0) / max(sum(uav_counts.values()), 1),
            "ttc_median_s": float(np.median(ttc)) if ttc.size else float("nan"),
            "ttc_p10_s": float(np.percentile(ttc, 10)) if ttc.size else float("nan"),
        })

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "conflict_sources.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    meta = {
        "rows": rows,
        "total_conflicts": total,
        "snapshots": n_snapshots,
        "conflicts_per_snapshot": total / max(n_snapshots, 1),
        "config": args.config,
        "num_uavs": sim.num_uavs,
        "grid_size_m": sim.grid_size,
        "density_preset": sim.density_preset,
        "cause_mix": sim.cause_mix,
        "label_mode": sim.label_mode,
        "lookahead_s": sim.lookahead_s,
        "split": args.split,
        "scenarios": args.scenarios,
        "duration_s": args.duration,
        "seed": sim.base_seed,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "machine": platform.node(),
    }
    with open(out / "conflict_sources.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"{'cause':20s} {'count':>7s} {'share':>7s} {'uav_share':>9s} {'ttc_med':>8s}")
    for r in rows:
        print(f"{r['cause']:20s} {r['count']:7d} {r['share']:7.1%} {r['uav_share']:9.1%} {r['ttc_median_s']:8.1f}")
    print(f"total={total} snapshots={n_snapshots} conflicts/snapshot={total / max(n_snapshots, 1):.2f}")
    print(f"written: {out / 'conflict_sources.csv'}")


if __name__ == "__main__":
    main()
