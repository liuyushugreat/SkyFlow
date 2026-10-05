"""Conflict detection evaluation metrics.

CDR (recall), FAR (1 - precision), F1, and inference latency.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch


@dataclass
class MetricResult:
    cdr: float          # Conflict Detection Rate = recall
    far: float          # False Alert Rate = 1 - precision
    f1: float
    precision: float
    recall: float
    latency_ms: float   # 95th-percentile wall-clock inference time
    latency_mean_ms: float
    num_pairs: int
    num_positives: int
    # S4: positives outside the candidate set (counted as FN) and per-stage timing
    num_missed_positives: int = 0
    stage_ms: Dict[str, float] = field(default_factory=dict)   # mean ms per stage
    stage_p95_ms: Dict[str, float] = field(default_factory=dict)
    # per-regime (hard/easy by time-to-conflict) and per-cause detection rates
    per_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        from dataclasses import asdict
        return asdict(self)


STAGES = ("graph_build", "gnn_forward", "pair_scoring")
CAUSE_NAMES = ("planned_crossing", "wind_deviation", "nonconforming", "priority_insertion", "noncooperative")


class ConflictMetrics:
    """Accumulates predictions across batches and computes final metrics.

    Positives that were never scored (because the candidate pre-filter
    dropped them) are passed via ``n_missed`` (optionally with their ttc /
    cause) and counted as false negatives, so CDR reflects the whole
    pipeline, not only the scorer."""

    def __init__(self, threshold: float = 0.42, regime_ttc_boundary_s: float = 15.0):
        self.threshold = threshold
        self.ttc_boundary = regime_ttc_boundary_s
        self.all_preds: List[np.ndarray] = []
        self.all_labels: List[np.ndarray] = []
        self.all_ttc: List[np.ndarray] = []
        self.all_cause: List[np.ndarray] = []
        self.latencies: List[float] = []
        self.n_missed = 0
        self.missed_ttc: List[np.ndarray] = []
        self.missed_cause: List[np.ndarray] = []
        self.stages: Dict[str, List[float]] = {s: [] for s in STAGES}

    def reset(self):
        self.__init__(self.threshold, self.ttc_boundary)

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        latency_ms: Optional[float] = None,
        n_missed: int = 0,
        stage_ms: Optional[Dict[str, float]] = None,
        ttc: Optional[torch.Tensor] = None,
        cause: Optional[torch.Tensor] = None,
        missed_ttc: Optional[torch.Tensor] = None,
        missed_cause: Optional[torch.Tensor] = None,
    ):
        p = preds.detach().cpu().numpy()
        self.all_preds.append(p)
        self.all_labels.append(labels.detach().cpu().numpy())
        self.all_ttc.append(ttc.detach().cpu().numpy() if ttc is not None else np.full(len(p), -1.0, np.float32))
        self.all_cause.append(cause.detach().cpu().numpy().astype(np.int16) if cause is not None else np.full(len(p), -1, np.int16))
        if latency_ms is not None:
            self.latencies.append(latency_ms)
        if stage_ms:
            for k, v in stage_ms.items():
                self.stages.setdefault(k, []).append(float(v))
        self.add_missed(n_missed, missed_ttc, missed_cause)

    def add_missed(self, n_missed: int, missed_ttc: Optional[torch.Tensor] = None,
                   missed_cause: Optional[torch.Tensor] = None):
        """Record positives of a snapshot that were not scored."""
        n_missed = int(n_missed)
        if n_missed <= 0:
            return
        self.n_missed += n_missed
        self.missed_ttc.append(missed_ttc.detach().cpu().numpy() if missed_ttc is not None
                               else np.full(n_missed, -1.0, np.float32))
        self.missed_cause.append(missed_cause.detach().cpu().numpy().astype(np.int16) if missed_cause is not None
                                 else np.full(n_missed, -1, np.int16))

    def _per_regime(self, preds, labels, ttc, cause) -> Dict[str, Dict[str, float]]:
        out: Dict[str, Dict[str, float]] = {}
        predicted_pos = preds >= self.threshold
        actual_pos = labels >= 0.5
        m_ttc = np.concatenate(self.missed_ttc) if self.missed_ttc else np.zeros(0, np.float32)
        m_cause = np.concatenate(self.missed_cause) if self.missed_cause else np.zeros(0, np.int16)

        # Regimes partition the *positives* (negatives have no TTC / cause),
        # so only recall-type quantities are defined per regime.
        def block(name, mask_scored, mask_missed):
            tp = int(np.sum(predicted_pos & actual_pos & mask_scored))
            fn = int(np.sum(~predicted_pos & actual_pos & mask_scored)) + int(np.sum(mask_missed))
            n_pos = tp + fn
            out[name] = {
                "cdr": tp / max(n_pos, 1),
                "num_positives": n_pos,
                "num_missed": int(np.sum(mask_missed)),
            }

        valid_ttc = ttc >= 0
        block("hard", valid_ttc & (ttc <= self.ttc_boundary), (m_ttc >= 0) & (m_ttc <= self.ttc_boundary))
        block("easy", valid_ttc & (ttc > self.ttc_boundary), (m_ttc >= 0) & (m_ttc > self.ttc_boundary))
        for code, name in enumerate(CAUSE_NAMES):
            if np.any(cause == code) or np.any(m_cause == code):
                block(f"cause:{name}", cause == code, m_cause == code)
        return out

    def compute(self) -> MetricResult:
        preds = np.concatenate(self.all_preds) if self.all_preds else np.zeros(0)
        labels = np.concatenate(self.all_labels) if self.all_labels else np.zeros(0)
        ttc = np.concatenate(self.all_ttc) if self.all_ttc else np.zeros(0, np.float32)
        cause = np.concatenate(self.all_cause) if self.all_cause else np.zeros(0, np.int16)

        predicted_pos = preds >= self.threshold
        actual_pos = labels >= 0.5

        tp = np.sum(predicted_pos & actual_pos)
        fp = np.sum(predicted_pos & ~actual_pos)
        fn = np.sum(~predicted_pos & actual_pos) + self.n_missed
        tn = np.sum(~predicted_pos & ~actual_pos)

        recall = tp / max(tp + fn, 1)
        precision = tp / max(tp + fp, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        far = fp / max(tp + fp, 1)

        if self.latencies:
            lat_95 = float(np.percentile(self.latencies, 95))
            lat_mean = float(np.mean(self.latencies))
        else:
            lat_95 = 0.0
            lat_mean = 0.0

        stage_ms = {k: float(np.mean(v)) for k, v in self.stages.items() if v}
        stage_p95 = {k: float(np.percentile(v, 95)) for k, v in self.stages.items() if v}

        return MetricResult(
            cdr=float(recall),
            far=float(far),
            f1=float(f1),
            precision=float(precision),
            recall=float(recall),
            latency_ms=lat_95,
            latency_mean_ms=lat_mean,
            num_pairs=len(preds),
            num_positives=int(np.sum(actual_pos)) + self.n_missed,
            num_missed_positives=self.n_missed,
            stage_ms=stage_ms,
            stage_p95_ms=stage_p95,
            per_regime=self._per_regime(preds, labels, ttc, cause),
        )


class RegimeMetrics:
    """Compute per-regime (easy/hard) CDR and FAR.

    Per paper Section 6.2: easy = TTC > 45s, pairwise, benign weather;
    hard = TTC <= 45s, multi-aircraft, or high-wind-variance cells.
    """

    def __init__(self, threshold: float = 0.42, ttc_boundary: float = 45.0):
        self.threshold = threshold
        self.ttc_boundary = ttc_boundary
        self.easy_preds: List[np.ndarray] = []
        self.easy_labels: List[np.ndarray] = []
        self.hard_preds: List[np.ndarray] = []
        self.hard_labels: List[np.ndarray] = []

    def reset(self):
        self.easy_preds.clear()
        self.easy_labels.clear()
        self.hard_preds.clear()
        self.hard_labels.clear()

    def update(
        self,
        preds: torch.Tensor,
        labels: torch.Tensor,
        is_hard: torch.Tensor,
    ):
        """is_hard: (N,) bool tensor indicating hard conflict regime."""
        p = preds.detach().cpu().numpy()
        l = labels.detach().cpu().numpy()
        h = is_hard.detach().cpu().numpy().astype(bool)
        if h.any():
            self.hard_preds.append(p[h])
            self.hard_labels.append(l[h])
        easy_mask = ~h
        if easy_mask.any():
            self.easy_preds.append(p[easy_mask])
            self.easy_labels.append(l[easy_mask])

    def compute(self) -> dict:
        result = {}
        for regime, p_list, l_list in [
            ("easy", self.easy_preds, self.easy_labels),
            ("hard", self.hard_preds, self.hard_labels),
        ]:
            if not p_list:
                result[regime] = {"cdr": 0.0, "far": 0.0}
                continue
            preds = np.concatenate(p_list)
            labels = np.concatenate(l_list)
            predicted_pos = preds >= self.threshold
            actual_pos = labels >= 0.5
            tp = np.sum(predicted_pos & actual_pos)
            fp = np.sum(predicted_pos & ~actual_pos)
            fn = np.sum(~predicted_pos & actual_pos)
            recall = tp / max(tp + fn, 1)
            far = fp / max(tp + fp, 1)
            result[regime] = {"cdr": float(recall), "far": float(far)}
        return result


def bonferroni_ttest(
    trgat_cdrs: List[float],
    baseline_cdrs: Dict[str, List[float]],
    alpha: float = 0.05,
) -> Dict[str, dict]:
    """Paired two-sided t-test with Bonferroni correction (Table 5 in paper)."""
    from scipy import stats

    n_comparisons = len(baseline_cdrs)
    corrected_alpha = alpha / max(n_comparisons, 1)
    results = {}

    trgat = np.array(trgat_cdrs)
    for name, bl_cdrs in baseline_cdrs.items():
        bl = np.array(bl_cdrs)
        delta = trgat - bl
        mean_delta = float(np.mean(delta))
        if len(delta) > 1 and np.std(delta) > 1e-12:
            t_stat, p_val = stats.ttest_rel(trgat, bl)
        else:
            t_stat, p_val = float("inf"), 0.0
        results[name] = {
            "delta_cdr": mean_delta,
            "t_stat": float(t_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < corrected_alpha),
            "corrected_alpha": corrected_alpha,
        }
    return results


class LatencyTimer:
    """Context manager for measuring inference latency."""

    def __init__(self):
        self._start = 0.0
        self.elapsed_ms = 0.0

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
