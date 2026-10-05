"""Load a finished task directory (results/<root>/<method>/seed<n>/) back
into an evaluator, for eval-only / robustness / scaling / attention scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch

from skyflow.baselines.registry import get_baseline
from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.experiments.baseline_trainer import evaluate_baseline
from skyflow.experiments.methods import METHODS
from skyflow.training.metrics import MetricResult
from skyflow.training.trainer import SkyFlowTrainer


@dataclass
class LoadedMethod:
    method: str
    kind: str
    seed: int
    cfg: SkyFlowConfig
    task_dir: Path
    evaluate: Callable[[List[Tuple[TKGSnapshot, torch.Tensor]]], MetricResult]
    trainer: Optional[SkyFlowTrainer] = None      # TR-GAT family
    model: Optional[object] = None                # baselines
    metrics: Optional[dict] = None                # metrics.json of the task
    checkpoint_epoch: Optional[int] = None


def task_dir(results_dir: str | Path, method: str, seed: int) -> Path:
    return Path(results_dir) / method / f"seed{seed}"


def list_tasks(results_dir: str | Path, method: Optional[str] = None, require_done: bool = True):
    """Yield (method, seed, dir) for finished tasks under results_dir."""
    root = Path(results_dir)
    if not root.exists():
        return
    for mdir in sorted(root.iterdir()):
        if not mdir.is_dir() or mdir.name not in METHODS:
            continue
        if method is not None and mdir.name != method:
            continue
        for sdir in sorted(mdir.glob("seed*")):
            if require_done and not (sdir / "DONE").exists():
                continue
            try:
                seed = int(sdir.name[4:])
            except ValueError:
                continue
            yield mdir.name, seed, sdir


def load_task(dir_: str | Path, device: torch.device, val_data=None) -> LoadedMethod:
    """Rebuild the model of a finished task from its config + checkpoint.

    Rule baselines have no checkpoint: CPA-Rule thresholds are restored from
    metrics.json (``training.rule_config``); if absent and ``val_data`` is
    given, they are re-fitted on it."""
    d = Path(dir_)
    cfg = SkyFlowConfig.from_yaml(d / "config.yaml")
    metrics = json.load(open(d / "metrics.json", encoding="utf-8")) if (d / "metrics.json").exists() else None
    method = metrics["method"] if metrics else d.parent.name
    seed = int(metrics["seed"]) if metrics else int(d.name[4:])
    spec = METHODS[method]

    if spec.kind == "trgat":
        trainer = SkyFlowTrainer(cfg, device=device)
        trainer.build_model()
        ckpt = torch.load(d / "best_model.pt", map_location=device, weights_only=False)
        trainer.model.load_state_dict(ckpt["model"])
        trainer.head.load_state_dict(ckpt["head"])
        trainer.model.eval()
        trainer.head.eval()
        return LoadedMethod(method, spec.kind, seed, cfg, d, trainer.evaluate, trainer=trainer,
                            metrics=metrics, checkpoint_epoch=ckpt.get("epoch"))

    model = get_baseline(spec.baseline_name, cfg, device)
    if spec.kind == "learned":
        ckpt = torch.load(d / "best_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        ep = ckpt.get("epoch")
    else:
        ep = None
        rule_cfg = (metrics or {}).get("training", {}).get("rule_config")
        if rule_cfg and hasattr(model, "h_thresh"):
            model.h_thresh = float(rule_cfg["h_thresh"])
            model.v_thresh = float(rule_cfg["v_thresh"])
        elif hasattr(model, "fit") and val_data is not None:
            model.fit(val_data)
    deterministic = spec.kind == "rule"

    def _eval(data):
        return evaluate_baseline(model, data, cfg, device, deterministic=deterministic)

    return LoadedMethod(method, spec.kind, seed, cfg, d, _eval, model=model, metrics=metrics, checkpoint_epoch=ep)
