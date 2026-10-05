#!/usr/bin/env python3
"""Attention vs. age-of-information (S7b).

Runs a trained TR-GAT over the test split and, for every relation type,
bins the per-edge attention coefficient alpha_ij by the edge's delta (AoI)
in --bin_s (0.25 s) bins.  Two versions are written:

  attn_mean        raw alpha_ij averaged over heads (depends on in-degree)
  attn_norm_mean   alpha_ij * deg_in(j) in that relation (1.0 = uniform attention),
                   which removes the trivial 1/degree effect

    python scripts/analyze_attention_aoi.py --checkpoint results/main/TR-GAT/seed42
Output: results/attention_vs_aoi.csv (+ .json provenance)
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
from skyflow.data.tkg_builder import relation_vocab
from skyflow.experiments.env_info import env_info
from skyflow.experiments.loader import load_task

FIELDS = ["relation", "relation_name", "layer", "delta_bin_lo_s", "delta_bin_hi_s", "n_edges",
          "attn_mean", "attn_std", "attn_norm_mean", "attn_norm_std", "mean_in_degree"]


@torch.no_grad()
def collect(trainer, data, layers):
    """Returns {(layer, r): (deltas, attn, attn_norm)} concatenated over all snapshots."""
    model = trainer.model
    model.eval()
    K = trainer.cfg.data.observation_window
    acc = defaultdict(lambda: ([], [], [], []))
    for window in trainer._group_into_windows(data, K):
        state = None
        for snapshot, _ in window:
            snapshot = trainer._to_device(snapshot)
            _, state = model(snapshot.node_features, snapshot.edge_indices, snapshot.edge_deltas, recurrent_state=state)
            N = snapshot.node_features.shape[0]
            for li in layers:
                layer = model.layers[li]
                for r, attn in getattr(layer, "last_attention", {}).items():
                    src, dst = snapshot.edge_indices[r]
                    deg = torch.bincount(dst, minlength=N).float()
                    a = attn.mean(dim=1)                          # average over heads
                    norm = a * deg[dst]
                    d, at, nm, dg = acc[(li, r)]
                    d.append(snapshot.edge_deltas[r].detach().cpu().numpy())
                    at.append(a.cpu().numpy())
                    nm.append(norm.cpu().numpy())
                    dg.append(deg[dst].cpu().numpy())
    return {k: tuple(np.concatenate(v) for v in vals) for k, vals in acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="TR-GAT task dir")
    ap.add_argument("--split", default="test")
    ap.add_argument("--bin_s", type=float, default=0.25)
    ap.add_argument("--max_delta_s", type=float, default=10.0)
    ap.add_argument("--layer", default="last", help="'last', 'all', or a 0-based index")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--out_csv", default="results/attention_vs_aoi.csv")
    args = ap.parse_args()

    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    lm = load_task(args.checkpoint, device)
    if lm.kind != "trgat":
        raise SystemExit("attention analysis needs a TR-GAT family checkpoint")
    if not lm.cfg.model.use_temporal:
        print("note: this checkpoint has use_temporal=False (attention cannot depend on delta)")
    n_layers = len(lm.trainer.model.layers)
    layers = [n_layers - 1] if args.layer == "last" else (list(range(n_layers)) if args.layer == "all" else [int(args.layer)])
    names = {v: k for k, v in relation_vocab(lm.cfg.leakage_free()).items()}
    data = get_split(lm.cfg, args.split, cache_dir=args.cache_dir, verbose=False)

    t0 = time.perf_counter()
    acc = collect(lm.trainer, data, layers)
    edges = np.arange(0.0, args.max_delta_s + args.bin_s, args.bin_s)
    rows = []
    for (li, r), (d, a, nm, dg) in sorted(acc.items()):
        d = np.clip(d, 0, args.max_delta_s - 1e-9)
        idx = np.digitize(d, edges) - 1
        for b in range(len(edges) - 1):
            m = idx == b
            if not np.any(m):
                continue
            rows.append({"relation": r, "relation_name": names.get(r, str(r)), "layer": li,
                         "delta_bin_lo_s": edges[b], "delta_bin_hi_s": edges[b + 1], "n_edges": int(m.sum()),
                         "attn_mean": float(a[m].mean()), "attn_std": float(a[m].std()),
                         "attn_norm_mean": float(nm[m].mean()), "attn_norm_std": float(nm[m].std()),
                         "mean_in_degree": float(dg[m].mean())})
    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})
    meta = {"checkpoint": str(lm.task_dir), "split": args.split, "snapshots": len(data), "bin_s": args.bin_s,
            "layers": layers, "n_rows": len(rows), "use_temporal": lm.cfg.model.use_temporal,
            "seconds": time.perf_counter() - t0, "env": env_info(device), "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(out.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)
    for r in sorted({row["relation"] for row in rows}):
        sub = [row for row in rows if row["relation"] == r and row["layer"] == layers[-1]]
        print(f"{names.get(r, r):16s} bins={len(sub):2d} edges={sum(x['n_edges'] for x in sub):7d} "
              f"norm-attn first/last bin: {sub[0]['attn_norm_mean']:.3f} / {sub[-1]['attn_norm_mean']:.3f}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
