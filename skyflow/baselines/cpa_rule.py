"""CPA-Rule baseline: state-based conflict detection as used in UTM/DAA.

For every candidate pair the *observed* (delayed, noisy) position and
velocity are linearly extrapolated over the look-ahead window W.  The pair
is flagged when there exists t ∈ [0, W] at which the horizontal separation
is below d_h **and** the vertical separation is below d_v simultaneously.
For linear motion both conditions hold on intervals of t, so the test is
an exact interval intersection (no time discretisation).

Thresholds:
  * ``threshold_mode="label"``      – use the label definition (10 m / 3 m).
  * ``threshold_mode="val_search"`` – grid-search (d_h, d_v) on a validation
    split and keep the pair with the best F1 (ties → smaller thresholds).

Deterministic; no parameters; a single run suffices.
"""

from __future__ import annotations

import itertools
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skyflow.data.tkg_builder import TKGSnapshot

THRESHOLD_MODES = ("label", "val_search")
DEFAULT_H_GRID: Tuple[float, ...] = (10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0)
DEFAULT_V_GRID: Tuple[float, ...] = (3.0, 5.0, 8.0, 10.0, 15.0)


def _pair_kinematics(snapshot: TKGSnapshot):
    feats = snapshot.node_features.detach().cpu().numpy()
    n = snapshot.num_uavs
    pairs = snapshot.conflict_pairs
    if pairs is None or pairs.size(1) == 0:
        return None
    src = pairs[0].detach().cpu().numpy()
    dst = pairs[1].detach().cpu().numpy()
    P, V = feats[:n, 0:3].astype(np.float64), feats[:n, 3:6].astype(np.float64)
    dp = P[dst] - P[src]
    dv = V[dst] - V[src]
    return dp, dv


def conflict_within_window(
    dp: np.ndarray, dv: np.ndarray, window_s: float, h_thresh: float, v_thresh: float
) -> np.ndarray:
    """Vectorised exact test: ∃ t∈[0,W] with |dp_h+dv_h t|<h and |dz+dvz t|<v."""
    W = float(window_s)
    eps = 1e-9

    # --- horizontal: a t² + b t + c < 0
    a = dv[:, 0] ** 2 + dv[:, 1] ** 2
    b = 2.0 * (dp[:, 0] * dv[:, 0] + dp[:, 1] * dv[:, 1])
    c = dp[:, 0] ** 2 + dp[:, 1] ** 2 - h_thresh ** 2
    lin = a < eps
    disc = b * b - 4.0 * a * c
    sq = np.sqrt(np.maximum(disc, 0.0))
    a_safe = np.where(lin, 1.0, a)
    h_lo = np.where(lin, np.where(c < 0, 0.0, np.inf), (-b - sq) / (2.0 * a_safe))
    h_hi = np.where(lin, np.where(c < 0, W, -np.inf), (-b + sq) / (2.0 * a_safe))
    h_empty = (~lin) & (disc <= 0)
    h_lo = np.where(h_empty, np.inf, h_lo)
    h_hi = np.where(h_empty, -np.inf, h_hi)

    # --- vertical: |dz + dvz t| < v
    dz, dvz = dp[:, 2], dv[:, 2]
    vlin = np.abs(dvz) < eps
    dvz_safe = np.where(vlin, 1.0, dvz)
    r1 = (-v_thresh - dz) / dvz_safe
    r2 = (v_thresh - dz) / dvz_safe
    v_lo = np.where(vlin, np.where(np.abs(dz) < v_thresh, 0.0, np.inf), np.minimum(r1, r2))
    v_hi = np.where(vlin, np.where(np.abs(dz) < v_thresh, W, -np.inf), np.maximum(r1, r2))

    lo = np.maximum.reduce([h_lo, v_lo, np.zeros_like(h_lo)])
    hi = np.minimum.reduce([h_hi, v_hi, np.full_like(h_hi, W)])
    return lo < hi


class CPARule:
    """Deterministic linear-extrapolation conflict detector."""

    def __init__(
        self,
        window_s: float = 30.0,
        h_thresh: float = 10.0,
        v_thresh: float = 3.0,
        threshold_mode: str = "label",
        h_grid: Sequence[float] = DEFAULT_H_GRID,
        v_grid: Sequence[float] = DEFAULT_V_GRID,
    ):
        if threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {threshold_mode!r}")
        self.window_s = float(window_s)
        self.h_thresh = float(h_thresh)
        self.v_thresh = float(v_thresh)
        self.threshold_mode = threshold_mode
        self.h_grid = tuple(h_grid)
        self.v_grid = tuple(v_grid)
        self.search_log: List[Dict] = []

    # ------------------------------------------------------------------ #
    def predict(self, snapshot: TKGSnapshot) -> torch.Tensor:
        """(P,) scores in {0, 1} for ``snapshot.conflict_pairs``."""
        kin = _pair_kinematics(snapshot)
        if kin is None:
            return torch.zeros(0)
        dp, dv = kin
        hit = conflict_within_window(dp, dv, self.window_s, self.h_thresh, self.v_thresh)
        return torch.tensor(hit.astype(np.float32))

    def fit(self, val_data: Iterable[Tuple[TKGSnapshot, torch.Tensor]]) -> Tuple[float, float]:
        """Grid-search thresholds on ``val_data`` (only in ``val_search`` mode).

        Missed positives (outside the candidate set) are included in the
        recall denominator so the chosen thresholds optimise pipeline F1."""
        if self.threshold_mode != "val_search":
            return self.h_thresh, self.v_thresh

        kins, labels, n_missed = [], [], 0
        for snap, lab in val_data:
            kin = _pair_kinematics(snap)
            n_missed += int(getattr(snap, "num_missed_positives", 0))
            if kin is None:
                continue
            kins.append(kin)
            labels.append(lab.detach().cpu().numpy() >= 0.5)
        if not kins:
            return self.h_thresh, self.v_thresh
        dp = np.concatenate([k[0] for k in kins])
        dv = np.concatenate([k[1] for k in kins])
        y = np.concatenate(labels)

        best = (-1.0, self.h_thresh, self.v_thresh)
        self.search_log = []
        for h, v in itertools.product(self.h_grid, self.v_grid):
            pred = conflict_within_window(dp, dv, self.window_s, h, v)
            tp = int(np.sum(pred & y))
            fp = int(np.sum(pred & ~y))
            fn = int(np.sum(~pred & y)) + n_missed
            prec = tp / max(tp + fp, 1)
            rec = tp / max(tp + fn, 1)
            f1 = 2 * prec * rec / max(prec + rec, 1e-12)
            self.search_log.append({"h": h, "v": v, "f1": f1, "precision": prec, "recall": rec})
            if f1 > best[0] + 1e-12:
                best = (f1, h, v)
        _, self.h_thresh, self.v_thresh = best
        return self.h_thresh, self.v_thresh

    def count_parameters(self) -> int:
        return 0

    def describe(self) -> Dict:
        return {
            "window_s": self.window_s,
            "h_thresh": self.h_thresh,
            "v_thresh": self.v_thresh,
            "threshold_mode": self.threshold_mode,
        }
