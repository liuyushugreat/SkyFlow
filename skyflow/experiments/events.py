"""Event-level (operational) evaluation (S8f).

The pair-snapshot metrics (CDR / FAR / F1) count every 1-Hz snapshot of a
conflict separately, so a 30 s conflict contributes up to 31 positives and a
detector that alerts on 15 of them scores CDR 0.5 even though the operator
was warned.  Operators reason in *events*:

  conflict event   maximal run of consecutive snapshots in which an unordered
                   UAV pair carries a positive label (same scenario; the
                   labels come from the unchanged 6-DoF look-ahead truth)
  detected         the detector alerted on >= 1 snapshot of the event
  lead time        time-to-conflict (s) at the FIRST alerted snapshot
  timely           detected with lead >= ``lead_s`` (denominator: events whose
                   first snapshot already had ttc >= lead_s, i.e. events for
                   which a timely alert was possible)
  alert episode    maximal run of consecutive alerted snapshots for a pair
                   (any label); TRUE if it overlaps >= 1 positive snapshot
  false episodes per UAV-hour  false alert episodes / (N_uav * hours observed)

Everything is computed from the per-snapshot scores of the already trained
checkpoints at their validation-selected thresholds; no label or model logic
is touched and the same code is applied to every method.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from skyflow.data.urbanair500 import CAUSES


@dataclass
class SnapshotScores:
    """Scores of one snapshot (numpy, CPU)."""
    index: int                 # position in the split (scenario order)
    pairs: np.ndarray          # (2, P) int64
    scores: np.ndarray         # (P,) float
    labels: np.ndarray         # (P,) float {0,1}
    ttc: np.ndarray            # (P,) float, -1 for negatives
    cause: np.ndarray          # (P,) int8, -1 for negatives
    num_uavs: int


def window_index_groups(n: int, K: int) -> List[List[int]]:
    """Same grouping as SkyFlowTrainer._group_into_windows, on indices."""
    groups = [list(range(s, s + K)) for s in range(0, n - K + 1, K)]
    if n >= K and n % K != 0:
        groups.append(list(range(n - K, n)))
    if not groups and n:
        groups.append(list(range(n)))
    return groups


def _np(t: Optional[torch.Tensor], dtype, size: int):
    if t is None:
        return np.full(size, -1, dtype=dtype)
    return t.detach().cpu().numpy().astype(dtype)


@torch.no_grad()
def iter_scores(lm, data) -> Iterator[SnapshotScores]:
    """Yield per-snapshot scores of a LoadedMethod (TR-GAT family keeps the
    GRU state across each K-window exactly as ``SkyFlowTrainer.evaluate``).
    Snapshots that appear twice (overlapping last window) are yielded once."""
    if lm.kind == "trgat":
        tr = lm.trainer
        tr.model.eval(); tr.head.eval()
        K = tr.cfg.data.observation_window
        seen = set()
        for group in window_index_groups(len(data), K):
            state = None
            for idx in group:
                snapshot, labels = data[idx]
                snapshot = tr._to_device(snapshot)
                emb, state = tr.model(snapshot.node_features, snapshot.edge_indices, snapshot.edge_deltas,
                                      recurrent_state=state)
                pairs = snapshot.conflict_pairs
                if idx in seen:
                    continue
                seen.add(idx)
                if pairs is None or pairs.size(1) == 0:
                    yield _empty(idx, snapshot.num_uavs)
                    continue
                preds = tr._score_pairs(snapshot, emb, state, pairs)
                yield _pack(idx, snapshot, preds, labels)
        return
    from skyflow.experiments.baseline_trainer import _to_device
    model = lm.model
    deterministic = lm.kind == "rule"
    if not deterministic:
        model.eval()
    device = _model_device(model)
    for idx, (snapshot, labels) in enumerate(data):
        snapshot = _to_device(snapshot, device)
        preds = model.predict(snapshot) if deterministic else model(snapshot)
        if preds.numel() == 0:
            yield _empty(idx, snapshot.num_uavs)
            continue
        yield _pack(idx, snapshot, preds, labels)


def _model_device(model) -> torch.device:
    params = getattr(model, "parameters", None)
    if callable(params):
        for p in params():
            return p.device
    return torch.device("cpu")


def _empty(idx: int, n: int) -> SnapshotScores:
    z = np.zeros(0)
    return SnapshotScores(idx, np.zeros((2, 0), np.int64), z, z, z, np.zeros(0, np.int8), n)


def _pack(idx: int, snapshot, preds: torch.Tensor, labels: torch.Tensor) -> SnapshotScores:
    P = snapshot.conflict_pairs.size(1)
    return SnapshotScores(
        idx, snapshot.conflict_pairs.detach().cpu().numpy().astype(np.int64),
        preds.detach().float().cpu().numpy().reshape(-1), labels.detach().float().cpu().numpy().reshape(-1),
        _np(snapshot.conflict_ttc, np.float32, P), _np(snapshot.conflict_cause, np.int8, P), snapshot.num_uavs,
    )


def method_threshold(lm) -> float:
    if lm.kind == "trgat":
        return float(getattr(lm.trainer, "threshold", lm.cfg.training.conflict_threshold))
    if lm.kind == "rule":
        return 0.5                         # rules output {0, 1}
    thr = getattr(lm.model, "threshold", None)
    return float(thr) if thr is not None else float(lm.cfg.training.conflict_threshold)


# --------------------------------------------------------------------------- event logic
KEY_SHIFT = 20


def pair_key(i: np.ndarray, j: np.ndarray) -> np.ndarray:
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return lo.astype(np.int64) * (1 << KEY_SHIFT) + hi.astype(np.int64)


def key_to_pair(key: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return key >> KEY_SHIFT, key & ((1 << KEY_SHIFT) - 1)


@dataclass
class EventTable:
    """Flat table of all scored pair-snapshots of one split, sorted by
    (scenario, pair, t).  Rows of the same pair at consecutive snapshots form
    a *track*; ``position`` is the 0-based index inside the track."""
    scenario: np.ndarray       # (R,) int
    t: np.ndarray              # (R,) int snapshot index within the scenario
    key: np.ndarray            # (R,) int64 unordered pair id
    label: np.ndarray          # (R,) bool
    score: np.ndarray          # (R,) float
    ttc: np.ndarray            # (R,) float
    cause: np.ndarray          # (R,) int8
    threshold: float
    num_uavs: int
    snapshots_per_scenario: int
    n_scenarios: int
    position: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        if self.position is None:
            self.position = _positions(self.scenario, self.key, self.t)

    @property
    def alert(self) -> np.ndarray:
        return self.score >= self.threshold

    def __len__(self):
        return int(self.score.size)

    @staticmethod
    def from_scores(scores: Sequence[SnapshotScores], threshold: float, snapshots_per_scenario: int) -> "EventTable":
        if snapshots_per_scenario <= 0:
            raise ValueError("snapshots_per_scenario must be positive")
        sc, tt, kk, ll, ss, cc, tc = [], [], [], [], [], [], []
        n_uavs = 0
        for s in scores:
            n_uavs = max(n_uavs, int(s.num_uavs))
            P = s.scores.size
            if P == 0:
                continue
            sc.append(np.full(P, s.index // snapshots_per_scenario, dtype=np.int64))
            tt.append(np.full(P, s.index % snapshots_per_scenario, dtype=np.int64))
            kk.append(pair_key(s.pairs[0], s.pairs[1]))
            ll.append(s.labels >= 0.5)
            ss.append(s.scores.astype(np.float64))
            cc.append(s.ttc.astype(np.float64))
            tc.append(s.cause.astype(np.int8))
        n_snap = (max(s.index for s in scores) + 1) if scores else 0
        n_scen = int(np.ceil(n_snap / snapshots_per_scenario)) if n_snap else 0
        cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
        scenario, t, key = cat(sc, np.int64), cat(tt, np.int64), cat(kk, np.int64)
        order = np.lexsort((t, key, scenario))
        return EventTable(scenario[order], t[order], key[order], cat(ll, bool)[order], cat(ss, np.float64)[order],
                          cat(cc, np.float64)[order], cat(tc, np.int8)[order], float(threshold), n_uavs,
                          snapshots_per_scenario, n_scen)

    def subset(self, keep: np.ndarray) -> "EventTable":
        """Rows selected by the boolean mask (order preserved; tracks recomputed)."""
        return EventTable(self.scenario[keep], self.t[keep], self.key[keep], self.label[keep], self.score[keep],
                          self.ttc[keep], self.cause[keep], self.threshold, self.num_uavs,
                          self.snapshots_per_scenario, self.n_scenarios)


def _positions(scenario: np.ndarray, key: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Position of each (sorted) row inside its track."""
    n = scenario.size
    if n == 0:
        return np.zeros(0, np.int64)
    new = np.ones(n, dtype=bool)
    new[1:] = (scenario[1:] != scenario[:-1]) | (key[1:] != key[:-1]) | (t[1:] != t[:-1] + 1)
    start = np.nonzero(new)[0]
    return np.arange(n) - np.repeat(start, np.diff(np.r_[start, n]))


def _runs(tab: EventTable, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group the rows selected by ``mask`` into maximal runs of consecutive
    snapshots of the same pair.  Returns (row indices in table order, run id)."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx, np.zeros(0, np.int64)
    new = np.ones(idx.size, dtype=bool)
    new[1:] = (idx[1:] != idx[:-1] + 1) | (tab.position[idx[1:]] == 0)
    return idx, np.cumsum(new) - 1


def _by_position(tab: EventTable) -> List[np.ndarray]:
    """Row indices grouped by track position 1, 2, ... (position 0 excluded);
    cached on the table because the argsort dominates on 10^7-row tables."""
    cached = getattr(tab, "_by_pos", None)
    if cached is not None:
        return cached
    pos = tab.position
    if pos.size == 0:
        groups = []
    else:
        # positions are < snapshots_per_scenario: 16-bit keys let numpy use radix sort
        order = np.argsort(pos.astype(np.uint16) if pos.max() < 65536 else pos, kind="stable")
        counts = np.bincount(pos)
        bounds = np.cumsum(counts)
        groups = [order[bounds[p - 1]:bounds[p]] for p in range(1, counts.size)]
    object.__setattr__(tab, "_by_pos", groups)
    return groups


def smooth_scores(tab: EventTable, alpha: float) -> np.ndarray:
    """Exponential moving average of the pair score along its track,
    s~_t = alpha * s_t + (1 - alpha) * s~_{t-1}, restarted whenever the pair
    re-enters the candidate set.  alpha = 1 returns the raw scores."""
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")
    s = tab.score.astype(np.float64).copy()
    if alpha >= 1.0:
        return s
    for rows in _by_position(tab):          # rows at position p depend on rows-1 (position p-1)
        s[rows] = alpha * tab.score[rows] + (1.0 - alpha) * s[rows - 1]
    return s


def hysteresis_alerts(tab: EventTable, scores: np.ndarray, theta_on: float, theta_off: float) -> np.ndarray:
    """Two-threshold alert logic along each track: an alert switches on when
    the score reaches theta_on and stays on while it remains >= theta_off
    (theta_off <= theta_on).  theta_off == theta_on is a plain threshold."""
    if theta_off > theta_on:
        raise ValueError("theta_off must not exceed theta_on")
    on = scores >= theta_on
    if theta_off < theta_on:
        hold = scores >= theta_off
        for rows in _by_position(tab):
            on[rows] |= on[rows - 1] & hold[rows]
    return on


def persistent_alerts(tab: EventTable, m: int, alert: Optional[np.ndarray] = None) -> np.ndarray:
    """M-of-M persistence filter (STCA-style confirmation): alerted at t only
    if alerted at t and the m-1 preceding snapshots of the same track."""
    alert = tab.alert if alert is None else alert
    if m <= 1:
        return alert
    idx, run = _runs(tab, alert)
    out = np.zeros_like(alert)
    if idx.size == 0:
        return out
    first = np.r_[0, np.nonzero(np.diff(run))[0] + 1]
    start_of_run = np.repeat(first, np.diff(np.r_[first, idx.size]))
    position = np.arange(idx.size) - start_of_run
    out[idx[position >= m - 1]] = True
    return out


@dataclass
class EventResult:
    persistence: int
    n_events: int
    n_detected: int
    event_cdr: float
    n_timely_possible: int
    n_timely: int
    timely_cdr: float
    lead_s: Dict[str, float]            # mean / median / p10 over detected events
    lead_frac: float                    # mean(lead / max_ttc) over detected events
    n_alert_episodes: int
    n_true_episodes: int
    n_false_episodes: int
    episode_precision: float
    false_episodes_per_uav_hour: float
    false_episode_duration_s: Dict[str, float]
    uav_hours: float
    per_cause: Dict[str, Dict[str, float]] = field(default_factory=dict)
    lead_s_list: Optional[np.ndarray] = None   # for figures; excluded from CSV
    threshold: float = float("nan")
    alpha: float = 1.0
    hysteresis: float = 0.0

    def to_row(self) -> Dict[str, float]:
        return {
            "persistence": self.persistence, "threshold": self.threshold, "alpha": self.alpha,
            "hysteresis": self.hysteresis,
            "n_events": self.n_events, "n_detected": self.n_detected, "event_cdr": self.event_cdr,
            "n_timely_possible": self.n_timely_possible, "n_timely": self.n_timely, "timely_cdr": self.timely_cdr,
            "lead_mean_s": self.lead_s["mean"], "lead_median_s": self.lead_s["median"], "lead_p10_s": self.lead_s["p10"],
            "lead_frac": self.lead_frac,
            "n_alert_episodes": self.n_alert_episodes, "n_true_episodes": self.n_true_episodes,
            "n_false_episodes": self.n_false_episodes, "episode_precision": self.episode_precision,
            "false_episodes_per_uav_hour": self.false_episodes_per_uav_hour,
            "false_episode_dur_mean_s": self.false_episode_duration_s["mean"],
            "false_episode_dur_median_s": self.false_episode_duration_s["median"],
            "uav_hours": self.uav_hours,
            "events_per_uav_hour": self.n_events / self.uav_hours if self.uav_hours > 0 else float("nan"),
        }


def _stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan")}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10))}


def event_metrics(tab: EventTable, lead_s: float = 10.0, snapshot_dt_s: float = 1.0,
                  persistence: int = 1, alert: Optional[np.ndarray] = None, with_cause: bool = True,
                  **tags) -> EventResult:
    """Event-level metrics for the given alert vector (default: raw threshold)."""
    alert = persistent_alerts(tab, int(persistence), alert)
    # ---- conflict events = runs of positive rows
    pos_idx, ev = _runs(tab, tab.label)
    n_events = int(ev.max() + 1) if ev.size else 0
    detected = np.zeros(n_events, dtype=bool)
    first_alert_ttc = np.full(n_events, np.nan)
    max_ttc = np.zeros(n_events)
    cause = np.full(n_events, -1, dtype=np.int64)
    if n_events:
        first_row = np.r_[0, np.nonzero(np.diff(ev))[0] + 1]     # earliest snapshot of each event
        max_ttc = tab.ttc[pos_idx[first_row]]
        cause = tab.cause[pos_idx[first_row]].astype(np.int64)
        alerted = alert[pos_idx]
        detected = np.bincount(ev[alerted], minlength=n_events) > 0
        a_rows = np.nonzero(alerted)[0]
        if a_rows.size:
            ev_a = ev[a_rows]
            first_a = np.r_[0, np.nonzero(np.diff(ev_a))[0] + 1]
            first_alert_ttc[ev_a[first_a]] = tab.ttc[pos_idx[a_rows[first_a]]]
    lead = first_alert_ttc[detected]
    timely_possible = max_ttc >= lead_s
    timely = detected & timely_possible & (np.nan_to_num(first_alert_ttc, nan=-1.0) >= lead_s)
    lead_frac = float(np.mean(lead / np.maximum(max_ttc[detected], 1e-9))) if lead.size else float("nan")

    # ---- alert episodes = runs of alerted rows (any label)
    al_idx, ep = _runs(tab, alert)
    n_ep = int(ep.max() + 1) if ep.size else 0
    if n_ep:
        true_ep = np.bincount(ep[tab.label[al_idx]], minlength=n_ep) > 0
        ep_len = np.bincount(ep, minlength=n_ep).astype(np.float64) * snapshot_dt_s
    else:
        true_ep = np.zeros(0, dtype=bool)
        ep_len = np.zeros(0)
    n_true = int(true_ep.sum())
    n_false = n_ep - n_true
    uav_hours = tab.num_uavs * tab.n_scenarios * tab.snapshots_per_scenario * snapshot_dt_s / 3600.0

    per_cause = {}
    if with_cause:
        for code, name in enumerate(CAUSES):
            m = cause == code
            if m.any():
                d = detected[m]
                per_cause[name] = {"n_events": int(m.sum()), "event_cdr": float(d.mean()),
                                   "timely_cdr": float(timely[m].sum() / max(timely_possible[m].sum(), 1)),
                                   "lead_median_s": float(np.median(first_alert_ttc[m][d])) if d.any() else float("nan")}

    return EventResult(
        persistence=int(persistence),
        n_events=n_events, n_detected=int(detected.sum()),
        event_cdr=float(detected.mean()) if n_events else float("nan"),
        n_timely_possible=int(timely_possible.sum()), n_timely=int(timely.sum()),
        timely_cdr=float(timely.sum() / timely_possible.sum()) if timely_possible.any() else float("nan"),
        lead_s=_stats(lead), lead_frac=lead_frac,
        n_alert_episodes=n_ep, n_true_episodes=n_true, n_false_episodes=n_false,
        episode_precision=float(n_true / n_ep) if n_ep else float("nan"),
        false_episodes_per_uav_hour=float(n_false / uav_hours) if uav_hours > 0 else float("nan"),
        false_episode_duration_s=_stats(ep_len[~true_ep]) if n_ep else _stats(np.zeros(0)),
        uav_hours=float(uav_hours), per_cause=per_cause, lead_s_list=lead,
        threshold=float(tags.get("threshold", tab.threshold)), alpha=float(tags.get("alpha", 1.0)),
        hysteresis=float(tags.get("hysteresis", 0.0)),
    )


# --------------------------------------------------------------------------- SOC curve (Kuchar 1996)
def filtered_alerts(tab: EventTable, threshold: float, alpha: float = 1.0, hysteresis: float = 0.0,
                    smoothed: Optional[np.ndarray] = None) -> np.ndarray:
    """Alert vector of the operational layer: EMA(alpha) -> hysteresis
    (on at threshold, off below threshold - hysteresis)."""
    s = smooth_scores(tab, alpha) if smoothed is None else smoothed
    return hysteresis_alerts(tab, s, float(threshold), float(threshold) - float(hysteresis))


def threshold_grid(tab: EventTable, n: int = 25, smoothed: Optional[np.ndarray] = None) -> np.ndarray:
    """Quantiles of the (smoothed) scores of positive rows: a grid that spans
    the whole operating range of this detector, dense where it matters."""
    s = tab.score if smoothed is None else smoothed
    pos = s[tab.label]
    if pos.size == 0:
        return np.array([tab.threshold])
    qs = np.linspace(0.02, 0.98, n)
    grid = np.unique(np.quantile(pos, qs))
    return np.unique(np.r_[grid, tab.threshold])


def soc_curve(tab: EventTable, thresholds: Sequence[float], alpha: float = 1.0, hysteresis: float = 0.0,
              lead_s: float = 10.0, snapshot_dt_s: float = 1.0) -> List[EventResult]:
    """System Operating Characteristic: event-level metrics for every
    threshold of the operational layer (alpha, hysteresis).  Rows that can
    never alert for any threshold of the grid are dropped first (exact)."""
    s = smooth_scores(tab, alpha)
    out = []
    for th in sorted(float(x) for x in thresholds):
        # progressive pruning: a negative row with smoothed score below the switch-off level can
        # neither alert nor hold an alert for this or any higher threshold
        keep = tab.label | (s >= th - float(hysteresis))
        if keep.sum() < 0.8 * keep.size:
            tab, s = tab.subset(keep), s[keep]
        a = hysteresis_alerts(tab, s, th, th - float(hysteresis))
        out.append(event_metrics(tab, lead_s=lead_s, snapshot_dt_s=snapshot_dt_s, alert=a, with_cause=False,
                                 threshold=th, alpha=alpha, hysteresis=hysteresis))
    return out


def select_operating_point(points: Sequence[EventResult], budget: float) -> Optional[EventResult]:
    """Among operating points whose false-episode rate is within ``budget``
    (per UAV-hour) return the one with the highest event CDR (ties: fewer
    false episodes, then higher lead).  None if no point satisfies the budget."""
    ok = [p for p in points if np.isfinite(p.false_episodes_per_uav_hour) and p.false_episodes_per_uav_hour <= budget]
    if not ok:
        return None
    return max(ok, key=lambda p: (p.event_cdr, -p.false_episodes_per_uav_hour,
                                  p.lead_s["median"] if np.isfinite(p.lead_s["median"]) else -1.0))


def evaluate_events(lm, data, snapshots_per_scenario: int, lead_s: float = 10.0,
                    snapshot_dt_s: float = 1.0, threshold: Optional[float] = None,
                    persistence: Sequence[int] = (1,)) -> List[EventResult]:
    """Score the split once (at the method's validation-selected threshold
    unless ``threshold`` is given) and return one EventResult per
    persistence value."""
    tab = score_table(lm, data, snapshots_per_scenario, threshold)
    return [event_metrics(tab, lead_s=lead_s, snapshot_dt_s=snapshot_dt_s, persistence=m) for m in persistence]


def score_table(lm, data, snapshots_per_scenario: int, threshold: Optional[float] = None) -> EventTable:
    thr = float(threshold) if threshold is not None else method_threshold(lm)
    return EventTable.from_scores(list(iter_scores(lm, data)), thr, snapshots_per_scenario)
