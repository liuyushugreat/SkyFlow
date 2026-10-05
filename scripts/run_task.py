#!/usr/bin/env python3
"""Run ONE (method, seed) task end-to-end and write its result directory (S7a).

    results/<root>/<method>/seed<n>/
        metrics.json        test metrics (CDR/FAR/F1, per-regime), training info, provenance
        config.yaml         exact config used
        history.json        per-epoch training curve
        best_model.pt       best checkpoint (trained methods)
        task.log
        DONE                written last; run_main --resume skips tasks that have it

Usage:
    python scripts/run_task.py --method TR-GAT --seed 42 --config configs/default.yaml --results_dir results/main
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import os
# Reduce allocator fragmentation (reserved 12 GB -> 8 GB on N=500) so that two
# tasks fit on one 24 GB GPU; must be set before CUDA initialises.  Override
# by exporting PYTORCH_CUDA_ALLOC_CONF yourself.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch

from skyflow.baselines.registry import get_baseline
from skyflow.config import SkyFlowConfig
from skyflow.data.cache import get_split
from skyflow.experiments.baseline_trainer import evaluate_baseline, train_baseline
from skyflow.experiments.env_info import env_info
from skyflow.experiments.methods import METHODS, method_config
from skyflow.models.input_norm import fit_input_norm
from skyflow.training.trainer import SkyFlowTrainer


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if hasattr(obj, "item") and callable(obj.item):
        try:
            return obj.item()
        except Exception:
            return str(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def run_task(method: str, seed: int, cfg_path: str, results_dir: str, device_str: str = "auto",
             epochs: int | None = None, cache_dir: str | None = None) -> Path:
    base = SkyFlowConfig.from_yaml(cfg_path)
    cfg = method_config(method, base)
    spec = METHODS[method]
    cfg.training.seed = seed
    out = Path(results_dir) / method / f"seed{seed}"
    out.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = str(out)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(out / "task.log", encoding="utf-8")], force=True)
    log = logging.getLogger("run_task")

    device = torch.device(device_str if device_str != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(cfg.training.tf32)
        torch.backends.cudnn.allow_tf32 = bool(cfg.training.tf32)
        torch.cuda.reset_peak_memory_stats(device)
    log.info(f"task {method} seed={seed} device={device} -> {out}")
    cfg.to_yaml(out / "config.yaml")

    t0 = time.perf_counter()
    train = val = None
    if spec.kind != "rule" or spec.baseline_name == "CPA-Rule":
        val = get_split(cfg, "val", device=torch.device("cpu"), cache_dir=cache_dir)
    if spec.kind in ("trgat", "learned"):
        train = get_split(cfg, "train", device=torch.device("cpu"), cache_dir=cache_dir)
    test = get_split(cfg, "test", device=torch.device("cpu"), cache_dir=cache_dir)
    data_seconds = time.perf_counter() - t0
    log.info(f"data ready in {data_seconds:.0f}s: train={len(train) if train else 0} "
             f"val={len(val) if val else 0} test={len(test)}")

    train_info = {}
    n_params = 0
    # seed BEFORE model construction so initial weights are reproducible per seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    normalize = bool(getattr(cfg.features, "normalize_inputs", True))
    if spec.kind == "trgat":
        trainer = SkyFlowTrainer(cfg, device=device)
        trainer.build_model()
        fit_input_norm(trainer.model, train, enabled=normalize)      # train-split stats, saved in the checkpoint
        n_params = trainer.model.count_parameters() + sum(p.numel() for p in trainer.head.parameters())
        train_info = trainer.train(train, val, seed=seed, output_dir=out, max_epochs=epochs)
        ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.head.load_state_dict(ckpt["head"])
        trainer.threshold = float(ckpt.get("threshold", trainer.threshold))
        test_metrics = trainer.evaluate(test)
        history = trainer.history
    elif spec.kind == "learned":
        model = get_baseline(spec.baseline_name, cfg, device)
        fit_input_norm(model, train, enabled=normalize)
        n_params = model.count_parameters()
        train_info = train_baseline(model, train, val, cfg, device, seed, out, max_epochs=epochs)
        history = train_info.pop("history", [])
        ckpt = torch.load(out / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.threshold = float(ckpt.get("threshold", cfg.training.conflict_threshold))
        test_metrics = evaluate_baseline(model, test, cfg, device, deterministic=False)
    else:  # rule
        model = get_baseline(spec.baseline_name, cfg, device)
        if hasattr(model, "fit") and val is not None:
            t_fit = time.perf_counter()
            model.fit(val)
            train_info = {"fit_seconds": time.perf_counter() - t_fit, "search_log": getattr(model, "search_log", [])}
        if hasattr(model, "describe"):
            train_info["rule_config"] = model.describe()
        test_metrics = evaluate_baseline(model, test, cfg, device, deterministic=True)
        history = []

    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else None
    metrics = {
        "method": method,
        "kind": spec.kind,
        "group": spec.group,
        "seed": seed,
        "test": test_metrics.to_dict(),
        "val_best": {k: v for k, v in train_info.items() if k in ("cdr", "far", "f1", "precision", "epoch", "per_regime")},
        "training": {k: v for k, v in train_info.items()
                     if k not in ("cdr", "far", "f1", "precision", "epoch", "per_regime", "seed")},
        "num_parameters": int(n_params),
        "threshold_mode": str(getattr(cfg.training, "threshold_mode", "val")),
        "test_threshold": float(test_metrics.threshold),
        "normalize_inputs": normalize,
        "data_seconds": data_seconds,
        "total_seconds": time.perf_counter() - t0,
        "peak_gpu_memory_gb": peak_mem,
        "amp": bool(cfg.training.amp),
        "tf32": bool(cfg.training.tf32),
        "env": env_info(device),
        "config_path": str(cfg_path),
        "dataset": {
            "num_uavs": cfg.data.num_uavs, "grid_size_m": cfg.data.grid_size_m,
            "train_scenarios": cfg.data.train_scenarios, "val_scenarios": cfg.data.val_scenarios,
            "test_scenarios": cfg.data.test_scenarios, "scenario_duration_s": cfg.data.scenario_duration_s,
            "test_snapshots": len(test),
        },
    }
    with open(out / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(metrics), f, indent=2)
    with open(out / "history.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(history), f, indent=2)
    (out / "DONE").write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")
    log.info(f"TEST {method} seed={seed}: CDR={test_metrics.cdr:.4f} FAR={test_metrics.far:.4f} "
             f"F1={test_metrics.f1:.4f} missed={test_metrics.num_missed_positives} "
             f"({metrics['total_seconds']:.0f}s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=sorted(METHODS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--results_dir", default="results/main")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=None, help="override max epochs")
    ap.add_argument("--cache_dir", default=None)
    args = ap.parse_args()
    run_task(args.method, args.seed, args.config, args.results_dir, args.device, args.epochs, args.cache_dir)


if __name__ == "__main__":
    main()
