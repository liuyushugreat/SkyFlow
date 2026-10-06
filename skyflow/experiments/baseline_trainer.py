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
from skyflow.training.io_utils import save_with_retry
from skyflow.training.losses import build_loss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer, MetricResult
from skyflow.training.trainer import build_scheduler, selection_score

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
                      device: torch.device, deterministic: bool, threshold: Optional[float] = None) -> MetricResult:
    """Threshold: explicit > ``model.threshold`` (val-selected, stored by
    train_baseline / restored by the loader) > config fixed threshold.
    Deterministic rules output {0,1}, so the threshold is irrelevant for them."""
    tc = cfg.training
    thr = threshold if threshold is not None else getattr(model, "threshold", None)
    thr = float(thr) if thr is not None else float(tc.conflict_threshold)
    metrics = ConflictMetrics(threshold=thr,
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
    # S8e: identical lr schedule to TR-GAT (warmup + cosine over the planned number of steps)
    steps_per_epoch = max(int(np.ceil(len(train_data) / batch)), 1)
    scheduler = build_scheduler(optimizer, getattr(tc, "scheduler", "warmup_cosine"),
                                tc.warmup_steps, n_epochs * steps_per_epoch)
    criterion = build_loss(getattr(tc, "loss", "focal"), tc.focal_gamma, getattr(tc, "focal_alpha", 0.75))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = output_dir / "best_model.pt"

    threshold_mode = str(getattr(tc, "threshold_mode", "val"))
    fixed_threshold = float(tc.conflict_threshold)
    model.threshold = fixed_threshold
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
            chunk = [train_data[k] for k in idx[s:s + batch]]
            # backward per snapshot (gradient accumulation): same effective
            # batch, but only one snapshot's graph is alive at a time
            total_loss, valid = 0.0, 0
            for snapshot, labels in chunk:
                snapshot = _to_device(snapshot, device)
                preds = model(snapshot)
                if preds.numel() == 0:
                    continue
                loss = criterion(preds, labels.to(device)) / len(chunk)
                loss.backward()
                total_loss += float(loss.detach()) * len(chunk)
                valid += 1
            if valid == 0:
                continue
            nn.utils.clip_grad_norm_(model.parameters(), tc.gradient_clip_norm)
            optimizer.step()
            scheduler.step()
            total += total_loss / valid
            n_steps += 1
        epochs_run = epoch + 1
        val = evaluate_baseline(model, val_data, cfg, device, deterministic=False, threshold=fixed_threshold)
        score, sel = selection_score(val, threshold_mode)
        rec = {"epoch": epochs_run, "train_loss": total / max(n_steps, 1), "val_f1": val.f1,
               "val_cdr": val.cdr, "val_far": val.far, "val_auprc": val.auprc, "val_best_f1": val.best_f1,
               "val_best_threshold": val.best_threshold, "val_selection": score,
               "epoch_seconds": time.perf_counter() - t_ep}
        history.append(rec)
        logger.info(f"Epoch {epochs_run}/{n_epochs} | Loss: {rec['train_loss']:.4f} | "
                    f"Val AUPRC: {val.auprc:.4f} | Val best-F1: {val.best_f1:.4f} @thr {val.best_threshold:.3f} "
                    f"(CDR {val.best_cdr:.3f}, FAR {val.best_far:.3f}) | F1@{fixed_threshold:.2f}: {val.f1:.4f}")
        if score > best_f1:
            best_f1, best_epoch, stale = score, epochs_run, 0
            model.threshold = float(sel["threshold"])
            best = {"cdr": sel["cdr"], "far": sel["far"], "f1": sel["f1"], "precision": 1.0 - sel["far"],
                    "epoch": epochs_run, "seed": seed, "per_regime": val.per_regime, "auprc": val.auprc,
                    "threshold": model.threshold, "selection_metric": sel["metric"], "selection_score": score}
            save_with_retry({"model": model.state_dict(), "epoch": epochs_run, "threshold": model.threshold,
                             "metrics": best, "config": cfg}, ckpt)
        else:
            stale += 1
        if patience > 0 and epochs_run >= min_epochs and stale >= patience and best_f1 > 0:   # same rule as TR-GAT
            logger.info(f"Early stopping at epoch {epochs_run} (best {threshold_mode} score {best_f1:.4f} @ {best_epoch})")
            break
    if not ckpt.exists():
        model.threshold = fixed_threshold
        save_with_retry({"model": model.state_dict(), "epoch": epochs_run, "threshold": fixed_threshold,
                         "metrics": {}, "config": cfg}, ckpt)
    best.update({"epochs_run": epochs_run, "best_epoch": best_epoch,
                 "train_seconds": time.perf_counter() - t0, "early_stopped": epochs_run < n_epochs,
                 "checkpoint": str(ckpt), "history": history})
    return best
