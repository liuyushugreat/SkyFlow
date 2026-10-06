"""SkyFlow training loop with warmup + cosine annealing and multi-seed evaluation."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR, SequentialLR

from skyflow.config import SkyFlowConfig
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import (
    ConflictScoringHead,
    build_pair_edge_features,
    pair_edge_feature_dim,
    plan_column,
)
from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.training.io_utils import save_with_retry
from skyflow.training.losses import FocalLoss, build_loss
from skyflow.training.metrics import ConflictMetrics, LatencyTimer, MetricResult

logger = logging.getLogger(__name__)

_OOM_ERRORS = (torch.cuda.OutOfMemoryError,) if hasattr(torch.cuda, "OutOfMemoryError") else (RuntimeError,)


def selection_score(m: MetricResult, threshold_mode: str) -> Tuple[float, Dict]:
    """Model-selection score on the validation split.

    'val'   : F1 at the F1-optimal validation threshold (threshold is then
              frozen and applied to test) - default.
    'fixed' : F1 at the fixed config threshold (legacy behaviour)."""
    if threshold_mode == "fixed":
        return float(m.f1), {"metric": "f1@fixed", "f1": m.f1, "cdr": m.cdr, "far": m.far, "threshold": m.threshold}
    if threshold_mode != "val":
        raise ValueError(f"threshold_mode must be 'val' or 'fixed', got {threshold_mode!r}")
    return float(m.best_f1), {"metric": "best_f1@val", "f1": m.best_f1, "cdr": m.best_cdr, "far": m.best_far,
                              "threshold": m.best_threshold}


def _is_oom(err: BaseException) -> bool:
    return isinstance(err, _OOM_ERRORS) or "out of memory" in str(err).lower()


def _build_warmup_cosine_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup for `warmup_steps`, then cosine decay to zero."""
    def warmup_fn(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        return 1.0

    warmup = LambdaLR(optimizer, lr_lambda=warmup_fn)
    cosine = CosineAnnealingLR(optimizer, T_max=max(total_steps - warmup_steps, 1))
    return SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_steps])


class SkyFlowTrainer:
    """End-to-end trainer for TR-GAT conflict detection."""

    def __init__(self, cfg: SkyFlowConfig, device: Optional[torch.device] = None):
        self.cfg = cfg

        if device is not None:
            self.device = device
        elif cfg.training.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(cfg.training.device)

        self.model: Optional[TRGAT] = None
        self.head: Optional[ConflictScoringHead] = None
        # S7a bookkeeping
        tc = cfg.training
        # windows whose graphs are kept alive per backward; gradients are
        # accumulated to the full batch_windows, so the maths is unchanged
        self.micro_batch = max(min(int(getattr(tc, "micro_batch_windows", 1)),
                                   int(getattr(tc, "batch_windows", 1))), 1)
        self.oom_adjustments: List[Dict] = []
        self.history: List[Dict] = []
        self.threshold: float = float(tc.conflict_threshold)   # replaced by the val-selected one after train()
        if self.device.type == "cuda":
            tf32 = bool(getattr(tc, "tf32", True))
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = tf32

    def build_model(self) -> Tuple[TRGAT, ConflictScoringHead]:
        mc = self.cfg.model
        self.model = TRGAT(
            node_feature_dim=self.cfg.uav_feature_dim(),
            embed_dim=mc.embed_dim,
            num_layers=mc.num_layers,
            num_heads=mc.num_heads,
            num_relations=self.cfg.num_relations(),
            temporal_dim=mc.temporal_dim,
            recurrent_dim=mc.recurrent_dim,
            dropout=mc.dropout,
            use_temporal=getattr(mc, "use_temporal", True),
            use_gating=getattr(mc, "use_gating", True),
            use_gru=getattr(mc, "use_gru", True),
        ).to(self.device)

        self.head = ConflictScoringHead(
            embed_dim=mc.embed_dim,
            recurrent_dim=mc.recurrent_dim,
            edge_feature_dim=pair_edge_feature_dim(self.cfg.features.pair_edge_features),
            dropout=mc.dropout,
        ).to(self.device)

        total_params = self.model.count_parameters() + sum(
            p.numel() for p in self.head.parameters() if p.requires_grad
        )
        logger.info(f"Model parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        return self.model, self.head

    def _group_into_windows(
        self,
        data: List[Tuple[TKGSnapshot, torch.Tensor]],
        K: int,
    ) -> List[List[Tuple[TKGSnapshot, torch.Tensor]]]:
        """Group consecutive snapshots into observation windows of size K."""
        windows = []
        for start in range(0, len(data) - K + 1, K):
            windows.append(data[start : start + K])
        if len(data) >= K and len(data) % K != 0:
            windows.append(data[-K:])
        if not windows and data:
            windows.append(data)
        return windows

    def train(
        self,
        train_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        val_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        seed: int = 42,
        output_dir: Optional[Path] = None,
        max_epochs: Optional[int] = None,
    ) -> Dict:
        """Train for one seed with early stopping on validation F1.

        Returns a dict with the best validation metrics plus training
        bookkeeping (epochs run, best epoch, wall time, OOM adjustments)."""
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.model is None:
            self.build_model()

        tc = self.cfg.training
        K = self.cfg.data.observation_window
        n_epochs = max_epochs if max_epochs is not None else tc.epochs
        batch_windows = max(int(getattr(tc, "batch_windows", 1)), 1)
        self.micro_batch = min(self.micro_batch, batch_windows)
        patience = int(getattr(tc, "early_stopping_patience", 0))
        min_epochs = int(getattr(tc, "min_epochs", 1))
        eval_every = max(int(getattr(tc, "eval_every", 1)), 1)
        use_amp = bool(getattr(tc, "amp", False)) and self.device.type == "cuda"

        params = list(self.model.parameters()) + list(self.head.parameters())
        optimizer = AdamW(params, lr=tc.learning_rate, weight_decay=tc.weight_decay)

        n_windows = len(self._group_into_windows(train_data, K))
        steps_per_epoch = max(int(np.ceil(n_windows / batch_windows)), 1)
        total_steps = n_epochs * steps_per_epoch
        scheduler = _build_warmup_cosine_scheduler(
            optimizer, warmup_steps=min(tc.warmup_steps, max(total_steps // 10, 1)), total_steps=total_steps
        )
        criterion = build_loss(getattr(tc, "loss", "focal"), tc.focal_gamma,
                               getattr(tc, "focal_alpha", 0.75))

        threshold_mode = str(getattr(tc, "threshold_mode", "val"))
        fixed_threshold = float(tc.conflict_threshold)
        self.threshold = fixed_threshold
        best_f1 = -1.0          # best selection score (val best-F1 in 'val' mode, F1@fixed in 'fixed')
        best_metrics: Dict = {}
        best_epoch = 0
        output_dir = Path(output_dir) if output_dir is not None else Path(self.cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = output_dir / "best_model.pt"
        self.history = []
        self.oom_adjustments = []

        def _save_checkpoint(metrics_dict, epoch):
            save_with_retry({
                "model": self.model.state_dict(),
                "head": self.head.state_dict(),
                "epoch": epoch,
                "threshold": self.threshold,
                "metrics": metrics_dict,
                "config": self.cfg,
            }, ckpt_path)

        def _window_loss(window) -> Tuple[torch.Tensor, int]:
            rec_state = None
            loss = 0.0
            valid = 0
            for snapshot, labels in window:
                snapshot = self._to_device(snapshot)
                labels = labels.to(self.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    node_emb, rec_state = self.model(
                        snapshot.node_features, snapshot.edge_indices, snapshot.edge_deltas,
                        recurrent_state=rec_state,
                    )
                pairs = snapshot.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    rec_state = rec_state.detach()
                    continue
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                    preds = self._score_pairs(snapshot, node_emb, rec_state, pairs)
                loss = loss + criterion(preds.float(), labels)
                valid += 1
                rec_state = rec_state.detach()
            return loss, valid

        t_start = time.perf_counter()
        epochs_run = 0
        stale = 0
        global_step = 0
        for epoch in range(n_epochs):
            self.model.train()
            self.head.train()
            epoch_loss = 0.0
            n_steps = 0
            t_epoch = time.perf_counter()

            windows = self._group_into_windows(train_data, K)
            np.random.shuffle(windows)

            for g in range(0, len(windows), batch_windows):
                group = windows[g:g + batch_windows]
                while True:
                    try:
                        optimizer.zero_grad(set_to_none=True)
                        group_loss = 0.0
                        n_valid_windows = 0
                        for m in range(0, len(group), self.micro_batch):
                            micro = group[m:m + self.micro_batch]
                            micro_loss = 0.0
                            micro_valid = 0
                            for window in micro:
                                wl, valid = _window_loss(window)
                                if valid > 0:
                                    micro_loss = micro_loss + wl / valid
                                    micro_valid += 1
                            if micro_valid > 0:
                                (micro_loss / len(group)).backward()      # accumulate -> same effective batch
                                group_loss += float(micro_loss.detach()) 
                                n_valid_windows += micro_valid
                        if n_valid_windows > 0:
                            nn.utils.clip_grad_norm_(params, tc.gradient_clip_norm)
                            optimizer.step()
                            epoch_loss += group_loss / n_valid_windows
                            n_steps += 1
                        scheduler.step()
                        global_step += 1
                        break
                    except Exception as err:            # noqa: BLE001
                        if not _is_oom(err) or self.micro_batch == 1:
                            raise
                        new_mb = max(self.micro_batch // 2, 1)
                        self.oom_adjustments.append({
                            "epoch": epoch + 1, "step": global_step,
                            "micro_batch_from": self.micro_batch, "micro_batch_to": new_mb,
                            "effective_batch_windows": batch_windows,
                        })
                        logger.warning(f"OOM: micro-batch {self.micro_batch} -> {new_mb} "
                                       f"(effective batch kept at {batch_windows} windows)")
                        self.micro_batch = new_mb
                        optimizer.zero_grad(set_to_none=True)
                        if self.device.type == "cuda":
                            torch.cuda.empty_cache()

            epochs_run = epoch + 1
            avg_loss = epoch_loss / max(n_steps, 1)
            record = {"epoch": epochs_run, "train_loss": avg_loss,
                      "epoch_seconds": time.perf_counter() - t_epoch,
                      "lr": optimizer.param_groups[0]["lr"], "micro_batch": self.micro_batch}

            if epochs_run % eval_every == 0 or epochs_run == n_epochs:
                val_metrics = self.evaluate(val_data, threshold=fixed_threshold)
                score, sel = selection_score(val_metrics, threshold_mode)
                record.update({"val_f1": val_metrics.f1, "val_cdr": val_metrics.cdr,
                               "val_far": val_metrics.far, "val_precision": val_metrics.precision,
                               "val_auprc": val_metrics.auprc, "val_best_f1": val_metrics.best_f1,
                               "val_best_threshold": val_metrics.best_threshold, "val_selection": score})
                logger.info(
                    f"Epoch {epochs_run}/{n_epochs} | Loss: {avg_loss:.4f} | "
                    f"Val AUPRC: {val_metrics.auprc:.4f} | Val best-F1: {val_metrics.best_f1:.4f} "
                    f"@thr {val_metrics.best_threshold:.3f} (CDR {val_metrics.best_cdr:.3f}, FAR {val_metrics.best_far:.3f}) | "
                    f"F1@{fixed_threshold:.2f}: {val_metrics.f1:.4f} | {record['epoch_seconds']:.1f}s"
                )
                if score > best_f1:
                    best_f1 = score
                    best_epoch = epochs_run
                    best_metrics = {
                        "cdr": sel["cdr"], "far": sel["far"], "f1": sel["f1"],
                        "precision": 1.0 - sel["far"], "latency_ms": val_metrics.latency_ms,
                        "epoch": epochs_run, "seed": seed, "per_regime": val_metrics.per_regime,
                        "auprc": val_metrics.auprc, "threshold": sel["threshold"],
                        "selection_metric": sel["metric"], "selection_score": score,
                    }
                    self.threshold = sel["threshold"]
                    _save_checkpoint(best_metrics, epochs_run)
                    stale = 0
                else:
                    stale += 1
            self.history.append(record)

            # never stop while the selection score has not left zero: with 0.16 %
            # positives the thresholded F1 is 0 for the first epochs although the loss falls
            if patience > 0 and epochs_run >= min_epochs and stale >= patience and best_f1 > 0:
                logger.info(f"Early stopping at epoch {epochs_run} (best {threshold_mode} score {best_f1:.4f} @ {best_epoch})")
                break

        if not ckpt_path.exists():
            best_metrics = {"cdr": 0, "far": 1, "f1": 0, "precision": 0, "latency_ms": 0, "epoch": epochs_run,
                            "seed": seed, "per_regime": {}, "auprc": 0.0, "threshold": self.threshold,
                            "selection_metric": threshold_mode, "selection_score": 0.0}
            _save_checkpoint(best_metrics, epochs_run)

        best_metrics = dict(best_metrics)
        best_metrics.update({
            "epochs_run": epochs_run,
            "best_epoch": best_epoch,
            "train_seconds": time.perf_counter() - t_start,
            "early_stopped": epochs_run < n_epochs,
            "oom_adjustments": list(self.oom_adjustments),
            "micro_batch_final": self.micro_batch,
            "effective_batch_windows": batch_windows,
            "checkpoint": str(ckpt_path),
        })
        return best_metrics

    @torch.no_grad()
    def evaluate(
        self,
        data: List[Tuple[TKGSnapshot, torch.Tensor]],
        threshold: Optional[float] = None,
    ) -> MetricResult:
        """Evaluate at ``threshold`` (default: the trainer's current threshold,
        i.e. the validation-selected one after training / checkpoint load)."""
        self.model.eval()
        self.head.eval()
        K = self.cfg.data.observation_window
        tc = self.cfg.training
        thr = float(threshold) if threshold is not None else float(getattr(self, "threshold", tc.conflict_threshold))
        metrics = ConflictMetrics(threshold=thr,
                                  regime_ttc_boundary_s=getattr(tc, "regime_ttc_boundary_s", 15.0))

        windows = self._group_into_windows(data, K)
        for window in windows:
            rec_state = None
            for snapshot, labels in window:
                snapshot = self._to_device(snapshot)
                labels = labels.to(self.device)

                t_gnn = LatencyTimer()
                with t_gnn:
                    node_emb, rec_state = self.model(
                        snapshot.node_features,
                        snapshot.edge_indices,
                        snapshot.edge_deltas,
                        recurrent_state=rec_state,
                    )

                pairs = snapshot.conflict_pairs
                if pairs is None or pairs.size(1) == 0:
                    metrics.add_missed(snapshot.num_missed_positives, snapshot.missed_ttc, snapshot.missed_cause)
                    continue

                t_score = LatencyTimer()
                with t_score:
                    preds = self._score_pairs(snapshot, node_emb, rec_state, pairs)

                stage_ms = {
                    "graph_build": snapshot.build_time_ms,
                    "gnn_forward": t_gnn.elapsed_ms,
                    "pair_scoring": t_score.elapsed_ms,
                }
                metrics.update(
                    preds, labels,
                    latency_ms=t_gnn.elapsed_ms + t_score.elapsed_ms,
                    n_missed=snapshot.num_missed_positives,
                    stage_ms=stage_ms,
                    ttc=snapshot.conflict_ttc, cause=snapshot.conflict_cause,
                    missed_ttc=snapshot.missed_ttc, missed_cause=snapshot.missed_cause,
                )

        return metrics.compute()

    def train_multi_seed(
        self,
        train_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        val_data: List[Tuple[TKGSnapshot, torch.Tensor]],
        test_data: List[Tuple[TKGSnapshot, torch.Tensor]],
    ) -> Dict:
        """Train across multiple seeds and report mean +/- std."""
        all_results = []
        seeds = self.cfg.training.seeds[:self.cfg.training.num_seeds]

        for seed_idx, seed in enumerate(seeds):
            logger.info(f"\n{'='*60}\nSeed {seed_idx+1}/{len(seeds)} (seed={seed})\n{'='*60}")

            self.model = None
            self.head = None
            self.build_model()

            best = self.train(train_data, val_data, seed=seed)

            ckpt = torch.load(
                Path(self.cfg.output_dir) / "best_model.pt",
                map_location=self.device,
                weights_only=False,
            )
            self.model.load_state_dict(ckpt["model"])
            self.head.load_state_dict(ckpt["head"])

            test_metrics = self.evaluate(test_data)
            result = {
                "seed": seed,
                "cdr": test_metrics.cdr,
                "far": test_metrics.far,
                "f1": test_metrics.f1,
                "precision": test_metrics.precision,
                "latency_ms": test_metrics.latency_ms,
            }
            all_results.append(result)
            logger.info(f"Seed {seed} test: CDR={result['cdr']:.4f}, F1={result['f1']:.4f}")

        summary = self._summarize_seeds(all_results)
        output_dir = Path(self.cfg.output_dir)
        with open(output_dir / "multi_seed_results.json", "w") as f:
            json.dump({"seeds": all_results, "summary": summary}, f, indent=2)

        return summary

    def _summarize_seeds(self, results: List[Dict]) -> Dict:
        metrics_keys = ["cdr", "far", "f1", "precision", "latency_ms"]
        summary = {}
        for key in metrics_keys:
            values = [r[key] for r in results]
            summary[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        return summary

    def _score_pairs(
        self,
        snapshot: TKGSnapshot,
        node_emb: torch.Tensor,
        rec_state: torch.Tensor,
        pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Eq. (6): score pairs with leakage-free edge features e_ij."""
        edge_feat = build_pair_edge_features(
            snapshot.node_features, pairs, snapshot.uav_aoi,
            mode=self.cfg.features.pair_edge_features,
            window_s=self.cfg.data.lookahead_seconds,
            plan_col0=plan_column(getattr(snapshot, "feature_names", None)),
        )
        return self.head(
            node_emb[pairs[0]], node_emb[pairs[1]],
            rec_state[pairs[0]], rec_state[pairs[1]],
            edge_feat=edge_feat,
        )

    def _to_device(self, snapshot: TKGSnapshot) -> TKGSnapshot:
        snapshot.node_features = snapshot.node_features.to(self.device)
        if snapshot.uav_aoi is not None:
            snapshot.uav_aoi = snapshot.uav_aoi.to(self.device)
        snapshot.edge_indices = {
            r: e.to(self.device) for r, e in snapshot.edge_indices.items()
        }
        snapshot.edge_deltas = {
            r: d.to(self.device) for r, d in snapshot.edge_deltas.items()
        }
        if snapshot.conflict_pairs is not None:
            snapshot.conflict_pairs = snapshot.conflict_pairs.to(self.device)
        return snapshot
