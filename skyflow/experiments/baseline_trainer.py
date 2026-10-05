"""Uniform training / evaluation for baselines (same early-stopping rule as
TR-GAT, same metric code, missed positives counted)."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.training.losses import build_loss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer, MetricResult

logger = logging.getLogger(__name__)


def _to_device(snapshot: TKGSnapshot, device: torch.device) -> TKGSnapshot:
    snapshot.node_features = snapshot.node_features.to(device)
    snapshot.edge_indices = {r: e.to(device) for r, e in snapshot.edge_indices.items()}
    snapshot.edge_deltas = {r: d.to(device) for r, d in snapshot.edge_deltas.items()}
    if snapshot.conflict_pairs is not None:
        snapshot.conflict_pairs = snapshot.conflict_pairs.to(device)
    if snapshot.uav_aoi is not None:
        snapshot.uav_aoi = snapshot.uav_aoi.to(device)
    return snapshot


@torch.no_grad()
def evaluate_baseline(model, data: List[Tuple[TKGSnapshot, torch.Tensor]], cfg: SkyFlowConfig,
                      device: torch.device, deterministic: bool) -> MetricResult:
    tc = cfg.training
    metrics = ConflictMetrics(threshold=tc.conflict_threshold,
                              regime_ttc_boundary_s=getattr(tc, "regime_ttc_boundary_s", 15.0))
    if not deterministic:
        model.eval()
    for snapshot, labels in data:
        snapshot = _to_device(snapshot, device)
        labels = labels.to(device)
        timer = LatencyTimer()
        with timer:
            preds = model.predict(snapshot) if deterministic else model(snapshot)
        if preds.numel() == 0:
            metrics.add_missed(snapshot.num_missed_positives, snapshot.missed_ttc, snapshot.missed_cause)
            continue
        metrics.update(
            preds.to(device), labels, latency_ms=timer.elapsed_ms,
            n_missed=snapshot.num_missed_positives,
            stage_ms={"graph_build": snapshot.build_time_ms, "gnn_forward": timer.elapsed_ms, "pair_scoring": 0.0},
            ttc=snapshot.conflict_ttc, cause=snapshot.conflict_cause,
            missed_ttc=snapshot.missed_ttc, missed_cause=snapshot.missed_cause,
        )
    return metrics.compute()


def train_baseline(model: nn.Module, train_data, val_data, cfg: SkyFlowConfig, device: torch.device,
                   seed: int, output_dir: Path, max_epochs: Optional[int] = None) -> Dict:
    """Same rule as SkyFlowTrainer.train: per-epoch val F1, patience, best checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    tc = cfg.training
    n_epochs = max_epochs if max_epochs is not None else tc.epochs
    patience = int(getattr(tc, "early_stopping_patience", 0))
    min_epochs = int(getattr(tc, "min_epochs", 1))
    batch = max(int(getattr(tc, "batch_windows", 1)), 1) * cfg.data.observation_window  # snapshots per step

    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    criterion = build_loss(getattr(tc, "loss", "focal"), tc.focal_gamma, getattr(tc, "focal_alpha", 0.75))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "best_model.pt"

    best_f1, best_epoch, best, stale = -1.0, 0, {}, 0
    history = []
    t0 = time.perf_counter()
    epochs_run = 0
    idx = np.arange(len(train_data))
    for epoch in range(n_epochs):
        model.train()
        np.random.shuffle(idx)
        t_ep = time.perf_counter()
        total, n_steps = 0.0, 0
        for s in range(0, len(idx), batch):
            optimizer.zero_grad(set_to_none=True)
            loss = 0.0
            valid = 0
            for k in idx[s:s + batch]:
                snapshot, labels = train_data[k]
                snapshot = _to_device(snapshot, device)
                preds = model(snapshot)
                if preds.numel() == 0:
                    continue
                loss = loss + criterion(preds, labels.to(device))
                valid += 1
            if valid == 0:
                continue
            (loss / valid).backward()
            nn.utils.clip_grad_norm_(model.parameters(), tc.gradient_clip_norm)
            optimizer.step()
            total += float(loss.detach()) / valid
            n_steps += 1
        epochs_run = epoch + 1
        val = evaluate_baseline(model, val_data, cfg, device, deterministic=False)
        rec = {"epoch": epochs_run, "train_loss": total / max(n_steps, 1), "val_f1": val.f1,
               "val_cdr": val.cdr, "val_far": val.far, "epoch_seconds": time.perf_counter() - t_ep}
        history.append(rec)
        logger.info(f"Epoch {epochs_run}/{n_epochs} | Loss: {rec['train_loss']:.4f} | "
                    f"Val CDR: {val.cdr:.4f} | Val F1: {val.f1:.4f} | Val FAR: {val.far:.4f}")
        if val.f1 > best_f1:
            best_f1, best_epoch, stale = val.f1, epochs_run, 0
            best = {"cdr": val.cdr, "far": val.far, "f1": val.f1, "precision": val.precision,
                    "epoch": epochs_run, "seed": seed, "per_regime": val.per_regime}
            torch.save({"model": model.state_dict(), "epoch": epochs_run, "metrics": best, "config": cfg}, ckpt)
        else:
            stale += 1
        if patience > 0 and epochs_run >= min_epochs and stale >= patience:
            logger.info(f"Early stopping at epoch {epochs_run} (best F1 {best_f1:.4f} @ {best_epoch})")
            break
    if not ckpt.exists():
        torch.save({"model": model.state_dict(), "epoch": epochs_run, "metrics": {}, "config": cfg}, ckpt)
    best.update({"epochs_run": epochs_run, "best_epoch": best_epoch,
                 "train_seconds": time.perf_counter() - t0, "early_stopped": epochs_run < n_epochs,
                 "checkpoint": str(ckpt), "history": history})
    return best
