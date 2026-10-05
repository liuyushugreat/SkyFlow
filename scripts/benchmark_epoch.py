#!/usr/bin/env python3
"""Benchmark ONE training epoch of a method on this machine (S7a / S8 input).

Measures: data preparation time (cache build or load), wall time per epoch,
validation time, peak GPU memory, mean GPU utilisation (nvidia-smi polled at
1 Hz during the epoch) and the data-preparation share of total time.
Writes ``results/benchmark/{hostname}[_tag].json`` so S8 can extrapolate the
full compute plan without guessing.

    python scripts/benchmark_epoch.py --config configs/default.yaml --method TR-GAT --epochs 1
"""

import argparse
import json
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")   # same as run_task.py

import numpy as np
import torch

from skyflow.baselines.registry import get_baseline
from skyflow.config import SkyFlowConfig
from skyflow.data.cache import cache_paths, get_split
from skyflow.experiments.baseline_trainer import train_baseline
from skyflow.experiments.env_info import env_info
from skyflow.experiments.methods import METHODS, method_config
from skyflow.training.trainer import SkyFlowTrainer


class GpuPoller(threading.Thread):
    """Polls nvidia-smi utilisation every ``period`` seconds until stopped."""

    def __init__(self, index: int = 0, period: float = 1.0):
        super().__init__(daemon=True)
        self.index, self.period = index, period
        self.samples = []
        self.mem_samples = []
        self._stop = threading.Event()

    def run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.index}", "--query-gpu=utilization.gpu,memory.used",
                     "--format=csv,noheader,nounits"], stderr=subprocess.DEVNULL, timeout=5).decode().strip()
                util, mem = out.split(",")
                self.samples.append(float(util))
                self.mem_samples.append(float(mem))
            except Exception:
                pass
            self._stop.wait(self.period)

    def stop(self):
        self._stop.set()
        self.join(timeout=5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", default="TR-GAT", choices=sorted(METHODS))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--cache_dir", default=None)
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--out_dir", default="results/benchmark")
    args = ap.parse_args()

    spec = METHODS[args.method]
    if spec.kind == "rule":
        raise SystemExit("benchmark_epoch only applies to trained methods")
    cfg = method_config(args.method, SkyFlowConfig.from_yaml(args.config))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.training.tf32)
        torch.cuda.reset_peak_memory_stats(device)

    # ---- data -------------------------------------------------------------
    cached_before = {s: cache_paths(cfg, s, args.cache_dir)[0].exists() for s in ("train", "val")}
    t0 = time.perf_counter()
    train = get_split(cfg, "train", cache_dir=args.cache_dir)
    t_train_data = time.perf_counter() - t0
    t1 = time.perf_counter()
    val = get_split(cfg, "val", cache_dir=args.cache_dir)
    t_val_data = time.perf_counter() - t1
    data_seconds = t_train_data + t_val_data
    n_pairs = int(sum(s.conflict_pairs.shape[1] for s, _ in train))
    n_pos = int(sum(float(l.sum()) for _, l in train))

    # ---- model + one (or a few) epochs -----------------------------------------
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir) / "_tmp" / args.method
    out_dir.mkdir(parents=True, exist_ok=True)
    poller = GpuPoller(device.index or 0) if device.type == "cuda" else None
    if poller:
        poller.start()
    t2 = time.perf_counter()
    if spec.kind == "trgat":
        trainer = SkyFlowTrainer(cfg, device=device)
        trainer.build_model()
        n_params = trainer.model.count_parameters() + sum(p.numel() for p in trainer.head.parameters())
        info = trainer.train(train, val, seed=args.seed, output_dir=out_dir, max_epochs=args.epochs)
        history = trainer.history
    else:
        model = get_baseline(spec.baseline_name, cfg, device)
        n_params = model.count_parameters()
        info = train_baseline(model, train, val, cfg, device, args.seed, out_dir, max_epochs=args.epochs)
        history = info.pop("history", [])
    train_seconds = time.perf_counter() - t2
    if poller:
        poller.stop()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    epoch_secs = [h.get("epoch_seconds", None) for h in history if h.get("epoch_seconds") is not None]
    # trainer.history records 'epoch_seconds' for the train pass only; the remainder is validation
    train_pass = float(np.mean(epoch_secs)) if epoch_secs else train_seconds / max(args.epochs, 1)
    per_epoch_total = train_seconds / max(len(history), 1)
    result = {
        "method": args.method,
        "config": args.config,
        "epochs_measured": len(history),
        "data": {
            "train_snapshots": len(train), "val_snapshots": len(val),
            "train_pairs": n_pairs, "train_positives": n_pos,
            "num_uavs": cfg.data.num_uavs, "grid_size_m": cfg.data.grid_size_m,
            "train_scenarios": cfg.data.train_scenarios, "scenario_duration_s": cfg.data.scenario_duration_s,
            "cached_before_run": cached_before,
            "train_data_seconds": t_train_data, "val_data_seconds": t_val_data,
        },
        "timing": {
            "data_seconds": data_seconds,
            "train_seconds_total": train_seconds,
            "seconds_per_epoch_total": per_epoch_total,       # train pass + validation
            "seconds_per_epoch_train_pass": train_pass,
            "seconds_per_epoch_validation": max(per_epoch_total - train_pass, 0.0),
            "data_share_of_total": data_seconds / max(data_seconds + train_seconds, 1e-9),
        },
        "gpu": {
            "peak_memory_allocated_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None,
            "peak_memory_reserved_gb": torch.cuda.max_memory_reserved(device) / 1e9 if device.type == "cuda" else None,
            "util_mean_pct": float(np.mean(poller.samples)) if poller and poller.samples else None,
            "util_p95_pct": float(np.percentile(poller.samples, 95)) if poller and poller.samples else None,
            "util_samples": len(poller.samples) if poller else 0,
            "smi_mem_used_max_mb": float(np.max(poller.mem_samples)) if poller and poller.mem_samples else None,
        },
        "training": {
            "num_parameters": int(n_params),
            "micro_batch_final": info.get("micro_batch_final"),
            "oom_adjustments": info.get("oom_adjustments", []),
            "batch_windows": cfg.training.batch_windows,
            "observation_window": cfg.data.observation_window,
            "amp": bool(cfg.training.amp), "tf32": bool(cfg.training.tf32),
            "val_f1_after": info.get("f1"),
        },
        "env": env_info(device),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    name = platform.node() + (f"_{args.tag}" if args.tag else "") + f"_{args.method}.json"
    with open(out / name, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    t = result["timing"]
    g = result["gpu"]
    print(f"\n{args.method}: {t['seconds_per_epoch_total']:.1f} s/epoch "
          f"(train pass {t['seconds_per_epoch_train_pass']:.1f} s, val {t['seconds_per_epoch_validation']:.1f} s); "
          f"data {t['data_seconds']:.0f} s ({100 * t['data_share_of_total']:.0f}% of total); "
          f"peak GPU mem {g['peak_memory_allocated_gb']} GB; GPU util mean {g['util_mean_pct']}%")
    print(f"written: {out / name}")


if __name__ == "__main__":
    main()
