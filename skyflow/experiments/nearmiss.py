"""Near-miss analysis of false alerts (S8f).

A pair-snapshot is labelled positive iff the true trajectories violate the
separation standard (h_sep, v_sep) at some point of the look-ahead window.
The label is a hard cut on the *normalised separation*

    rho = min_k  max( d_h(k) / h_sep ,  d_v(k) / v_sep ),   k in the window,

which is < 1 exactly for positives.  "False" alerts with rho just above 1
are near misses, not nonsense; this module re-integrates the deterministic
test truth (same scenario seeds, labels untouched) and reports the rho
distribution of false alert rows / episodes against a random negative
sample, plus a consistency check that the labels of the cached split are
reproduced (positives rho < 1, negatives rho >= 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from skyflow.experiments.events import EventTable, _runs, key_to_pair

RHO_LEVELS = (1.0, 1.5, 2.0, 3.0)


def normalised_separation(positions: np.ndarray, epoch: int, i: np.ndarray, j: np.ndarray,
                          horizon_epochs: int, h_sep: float, v_sep: float) -> np.ndarray:
    """rho for pairs (i, j) at ``epoch`` over [epoch, epoch + horizon_epochs]
    of the true trajectory array ``positions`` (T_total, N, 3)."""
    end = min(epoch + horizon_epochs, positions.shape[0] - 1)
    P = positions[epoch:end + 1]
    if i.size == 0:
        return np.zeros(0)
    dp = P[:, j, :] - P[:, i, :]                       # (T, P, 3)
    dh = np.hypot(dp[..., 0], dp[..., 1]) / h_sep
    dv = np.abs(dp[..., 2]) / v_sep
    return np.maximum(dh, dv).min(axis=0)


@dataclass
class NearMissResult:
    rho_false_rows: np.ndarray
    rho_false_episodes: np.ndarray
    rho_negative_sample: np.ndarray
    rho_positive_check: np.ndarray
    n_positive_checked: int
    n_negative_checked: int
    positive_mismatch: int      # positives with rho >= 1 (should be 0)
    negative_mismatch: int      # negatives with rho < 1 (should be 0)
    levels: Sequence[float] = field(default_factory=lambda: RHO_LEVELS)

    def rows(self) -> List[Dict[str, float]]:
        out = []
        for name, arr in (("false_rows", self.rho_false_rows), ("false_episodes", self.rho_false_episodes),
                          ("negative_sample", self.rho_negative_sample), ("positive_check", self.rho_positive_check)):
            r = {"group": name, "n": int(arr.size),
                 "rho_median": float(np.median(arr)) if arr.size else float("nan"),
                 "rho_mean": float(arr.mean()) if arr.size else float("nan")}
            for lv in self.levels:
                r[f"frac_lt_{lv:g}"] = float((arr < lv).mean()) if arr.size else float("nan")
            out.append(r)
        return out


def near_miss_analysis(tab: EventTable, alert: np.ndarray, logs: Sequence, epoch_step: int, horizon_epochs: int,
                       h_sep: float, v_sep: float, n_negative_sample: int = 200_000, seed: int = 0,
                       levels: Sequence[float] = RHO_LEVELS) -> NearMissResult:
    """``logs[s]`` is the truth log of test scenario s (``positions``
    (T_total, N, 3)); snapshot t of a scenario is epoch t * epoch_step."""
    rng = np.random.RandomState(seed)
    neg = ~tab.label
    false_rows = np.nonzero(alert & neg)[0]
    neg_rows = np.nonzero(neg)[0]
    if neg_rows.size > n_negative_sample:
        neg_rows = np.sort(rng.choice(neg_rows, n_negative_sample, replace=False))
    pos_rows = np.nonzero(tab.label)[0]
    if pos_rows.size > n_negative_sample:
        pos_rows = np.sort(rng.choice(pos_rows, n_negative_sample, replace=False))

    def rho_of(rows: np.ndarray) -> np.ndarray:
        out = np.full(rows.size, np.nan)
        if rows.size == 0:
            return out
        sc, tt = tab.scenario[rows], tab.t[rows]
        i, j = key_to_pair(tab.key[rows])
        for s in np.unique(sc):
            log = logs[int(s)]
            m_s = sc == s
            for t in np.unique(tt[m_s]):
                m = m_s & (tt == t)
                out[m] = normalised_separation(log.positions, int(t) * epoch_step, i[m], j[m],
                                               horizon_epochs, h_sep, v_sep)
        return out

    rho_false = rho_of(false_rows)
    rho_neg = rho_of(neg_rows)
    rho_pos = rho_of(pos_rows)
    # false episodes = alert runs without any positive row; rho of an episode = min over its rows
    al_idx, ep = _runs(tab, alert)
    rho_ep = np.zeros(0)
    if al_idx.size:
        n_ep = int(ep.max() + 1)
        true_ep = np.bincount(ep[tab.label[al_idx]], minlength=n_ep) > 0
        rho_row = np.full(len(tab), np.inf)
        rho_row[false_rows] = rho_false
        ep_min = np.full(n_ep, np.inf)
        np.minimum.at(ep_min, ep, rho_row[al_idx])
        rho_ep = ep_min[~true_ep]
    return NearMissResult(
        rho_false_rows=rho_false, rho_false_episodes=rho_ep, rho_negative_sample=rho_neg, rho_positive_check=rho_pos,
        n_positive_checked=int(pos_rows.size), n_negative_checked=int(neg_rows.size),
        positive_mismatch=int((rho_pos >= 1.0).sum()), negative_mismatch=int((rho_neg < 1.0).sum()), levels=levels,
    )
