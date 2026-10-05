#!/usr/bin/env python3
"""Pre-build and cache TKG snapshots for train/val/test (S7a).

Usage:
    python scripts/build_cache.py --config configs/default.yaml
    python scripts/build_cache.py --config configs/smoke.yaml --splits train val
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skyflow.config import SkyFlowConfig
from skyflow.data.cache import cache_paths, cache_size_bytes, get_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--cache_dir", default=None, help="override data.cache_dir")
    ap.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = ap.parse_args()

    cfg = SkyFlowConfig.from_yaml(args.config)
    cache_dir = args.cache_dir or cfg.data.cache_dir
    t0 = time.perf_counter()
    for split in args.splits:
        pt, meta = cache_paths(cfg, split, cache_dir)
        if args.force and pt.exists():
            pt.unlink()
            if meta.exists():
                meta.unlink()
        data = get_split(cfg, split, cache_dir=cache_dir)
        n_pos = int(sum(float(l.sum()) for _, l in data))
        n_pairs = int(sum(s.conflict_pairs.shape[1] for s, _ in data))
        print(f"  {split:5s}: {len(data):5d} snapshots, {n_pairs:,} pairs, {n_pos:,} positives "
              f"({n_pos / max(n_pairs, 1):.3%}), file {pt.stat().st_size / 1e6:.1f} MB")
        del data
    total = cache_size_bytes(cache_dir)
    print(f"cache total: {total / 1e9:.2f} GB in {Path(cache_dir).resolve()}  "
          f"({time.perf_counter() - t0:.0f} s)")


if __name__ == "__main__":
    main()
