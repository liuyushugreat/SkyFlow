#!/usr/bin/env python3
"""Intent-conformance gate analysis (S8e).

Runs a trained TR-GAT (with ``model.use_conformance_gate``) over a split and
summarises the gate value w_i (how far the model trusts the filed plan of
UAV i) per group:

  uav:all / uav:with_plan            per-UAV gate values over all snapshots
  pair:negative / pair:positive      per scored pair, w_pair = min(w_i, w_j)
  pair:cause:<name>                  positives split by the injected conflict cause

A gate that has learned something should trust planned-crossing aircraft
more than non-conforming / wind-drifted ones (whose plans mislead).

    python scripts/analyze_gate.py --checkpoint results/main/TR-GAT/seed42
Output: results/gate_by_cause.csv (+ .json provenance)
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

from skyflow.data.cache import get_split
from skyflow.data.urbanair500 import CAUSES
from skyflow.experiments.env_info import env_info
from skyflow.experiments.loader import load_task

FIELDS = ["group", "n", "w_mean", "w_std", "w_median", "w_p10", "w_p90"]


@torch.no_grad()
def collect(trainer, data):
    model, head = trainer.model, trainer.head
    model.eval(); head.eval()
    K = trainer.cfg.data.observation_window
    groups = defaultdict(list)
    for window in trainer._group_into_windows(data, K):
        state = None
        for snapshot, labels in window:
            snapshot = trainer._to_device(snapshot)
            emb, state = model(snapshot.node_features, snapshot.edge_indices, snapshot.edge_deltas,
                               recurrent_state=state)
            pairs = snapshot.conflict_pairs
            if pairs is None or pairs.size(1) == 0:
                continue
            trainer._score_pairs(snapshot, emb, state, pairs)
            w = head.last_gate.float().cpu().numpy()
            n = snapshot.num_uavs
            w = w[:n]
            groups["uav:all"].append(w)
            has_plan = w > 0.0
            groups["uav:with_plan"].append(w[has_plan])
            i, j = pairs[0].cpu().numpy(), pairs[1].cpu().numpy()
            wp = np.minimum(w[i], w[j])
            y = labels.cpu().numpy() >= 0.5
            groups["pair:negative"].append(wp[~y])
            groups["pair:positive"].append(wp[y])
            cause = snapshot.conflict_cause
            if cause is not None:
                c = cause.cpu().numpy()
                for code, name in enumerate(CAUSES):
                    m = y & (c == code)
                    if m.any():
                        groups[f"pair:cause:{name}"].append(wp[m])
    return {k: np.concatenate(v) for k, v in groups.items() if v}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="TR-GAT task dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out_csv", default="results/gate_by_cause.csv")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    lm = load_task(args.checkpoint, device)
    if lm.kind != "trgat" or not getattr(lm.trainer.head, "conformance_gate", False):
        raise SystemExit("gate analysis needs a TR-GAT checkpoint trained with model.use_conformance_gate")
    data = get_split(lm.cfg, args.split, cache_dir=args.cache_dir, verbose=False)
    t0 = time.perf_counter()
    groups = collect(lm.trainer, data)
    rows = []
    for g, w in groups.items():
        rows.append({"group": g, "n": int(w.size), "w_mean": float(w.mean()), "w_std": float(w.std()),
                     "w_median": float(np.median(w)), "w_p10": float(np.percentile(w, 10)),
                     "w_p90": float(np.percentile(w, 90))})
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=FIELDS)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    meta = {"checkpoint": str(lm.task_dir), "split": args.split, "snapshots": len(data),
            "seconds": time.perf_counter() - t0, "env": env_info(device), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    for row in rows:
        print(f"{row['group']:28s} n={row['n']:9d} mean={row['w_mean']:.3f} median={row['w_median']:.3f} "
              f"p10/p90={row['w_p10']:.3f}/{row['w_p90']:.3f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
