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


def _uav_state(snapshot: TKGSnapshot, aoi_sync: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Observed (P, V) of the UAV rows; with ``aoi_sync`` every report is first
    dead-reckoned to the common epoch with its own age (P + V * AoI), the same
    synchronisation the learned scorers receive in the ``*_sync`` pair modes."""
    feats = snapshot.node_features.detach().cpu().numpy()
    n = snapshot.num_uavs
    P, V = feats[:n, 0:3].astype(np.float64), feats[:n, 3:6].astype(np.float64)
    aoi = getattr(snapshot, "uav_aoi", None)
    if aoi_sync and aoi is not None:
        age = aoi.detach().cpu().numpy().astype(np.float64)[:n]
        P = P + V * age[:, None]
    return P, V


def _pair_kinematics(snapshot: TKGSnapshot, aoi_sync: bool = False):
    pairs = snapshot.conflict_pairs
    if pairs is None or pairs.size(1) == 0:
        return None
    src = pairs[0].detach().cpu().numpy()
    dst = pairs[1].detach().cpu().numpy()
    P, V = _uav_state(snapshot, aoi_sync)
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
        aoi_sync: bool = False,
    ):
        if threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"threshold_mode must be one of {THRESHOLD_MODES}, got {threshold_mode!r}")
        self.aoi_sync = bool(aoi_sync)      # S8e: dead-reckon each report to the common epoch first
        self.window_s = float(window_s)
        self.h_thresh = float(h_thresh)
        self.v_thresh = float(v_thresh)
        self.threshold_mode = threshold_mode
        self.h_grid = tuple(h_grid)
        self.v_grid = tuple(v_grid)
        self.search_log: List[Dict] = []

    # ------------------------------------------------------------------ #
    def _segments(self, snapshot: TKGSnapshot) -> Optional[List[Tuple[np.ndarray, np.ndarray, float]]]:
        """Piecewise-linear relative motion of every candidate pair as a list of
        (dp, dv, duration) segments. The plain rule has one segment: observed
        relative state extrapolated over the whole window."""
        kin = _pair_kinematics(snapshot, self.aoi_sync)
        if kin is None:
            return None
        return [(kin[0], kin[1], self.window_s)]

    @staticmethod
    def _hit(segments: List[Tuple[np.ndarray, np.ndarray, float]], h: float, v: float) -> np.ndarray:
        hit = None
        for dp, dv, dur in segments:
            s = conflict_within_window(dp, dv, dur, h, v)
            hit = s if hit is None else (hit | s)
        return hit

    def predict(self, snapshot: TKGSnapshot) -> torch.Tensor:
        """(P,) scores in {0, 1} for ``snapshot.conflict_pairs``."""
        segs = self._segments(snapshot)
        if segs is None:
            return torch.zeros(0)
        hit = self._hit(segs, self.h_thresh, self.v_thresh)
        return torch.tensor(hit.astype(np.float32))

    def fit(self, val_data: Iterable[Tuple[TKGSnapshot, torch.Tensor]]) -> Tuple[float, float]:
        """Grid-search thresholds on ``val_data`` (only in ``val_search`` mode).

        Missed positives (outside the candidate set) are included in the
        recall denominator so the chosen thresholds optimise pipeline F1."""
        if self.threshold_mode != "val_search":
            return self.h_thresh, self.v_thresh

        per_seg: List[List[Tuple[np.ndarray, np.ndarray]]] = []
        durations: List[float] = []
        labels, n_missed = [], 0
        for snap, lab in val_data:
            segs = self._segments(snap)
            n_missed += int(getattr(snap, "num_missed_positives", 0))
            if segs is None:
                continue
            if not per_seg:
                per_seg = [[] for _ in segs]
                durations = [d for _, _, d in segs]
            for k, (dp, dv, _) in enumerate(segs):
                per_seg[k].append((dp, dv))
            labels.append(lab.detach().cpu().numpy() >= 0.5)
        if not per_seg:
            return self.h_thresh, self.v_thresh
        segments = [
            (np.concatenate([s[0] for s in seg]), np.concatenate([s[1] for s in seg]), durations[k])
            for k, seg in enumerate(per_seg)
        ]
        y = np.concatenate(labels)

        best = (-1.0, self.h_thresh, self.v_thresh)
        self.search_log = []
        for h, v in itertools.product(self.h_grid, self.v_grid):
            pred = self._hit(segments, h, v)
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
            "aoi_sync": self.aoi_sync,
        }


PLAN_FEATURE_PREFIXES = ("pl10", "pl20", "pl30")


class PlanCPARule(CPARule):
    """Plan-aware CPA rule (S8d): the same interval test applied to the
    *filed-plan* trajectory instead of linear extrapolation.

    Each UAV's planned path is the polyline through its observed position and
    the planned positions at +10/+20/+30 s read from the node features
    (``pl10_*``, ``pl20_*``, ``pl30_*`` - the same plan context the learned
    models receive).  UAVs without plan context (non-cooperative; all-zero
    offsets) fall back to linear extrapolation of the observed velocity.
    Separation is tested exactly on every 10 s segment.
    """

    def _segments(self, snapshot: TKGSnapshot):
        pairs = snapshot.conflict_pairs
        if pairs is None or pairs.size(1) == 0:
            return None
        names = list(getattr(snapshot, "feature_names", None) or [])
        if not all(f"{p}_dx" in names for p in PLAN_FEATURE_PREFIXES):
            raise ValueError("PlanCPARule needs plan-context features (features.plan_context: true)")
        feats = snapshot.node_features.detach().cpu().numpy()
        n = snapshot.num_uavs
        P, V = _uav_state(snapshot, self.aoi_sync)
        src = pairs[0].detach().cpu().numpy()
        dst = pairs[1].detach().cpu().numpy()

        n_seg = len(PLAN_FEATURE_PREFIXES)
        dur = self.window_s / n_seg
        knots = [P]
        for k, pfx in enumerate(PLAN_FEATURE_PREFIXES):
            c = names.index(f"{pfx}_dx")
            off = feats[:n, c:c + 3].astype(np.float64)
            no_plan = np.linalg.norm(off, axis=1) < 1e-6
            off = np.where(no_plan[:, None], V * (dur * (k + 1)), off)   # fallback: linear extrapolation
            knots.append(P + off)
        segs = []
        for k in range(n_seg):
            q0, q1 = knots[k], knots[k + 1]
            seg_v = (q1 - q0) / dur
            dp = q0[dst] - q0[src]
            dv = seg_v[dst] - seg_v[src]
            segs.append((dp, dv, dur))
        return segs
