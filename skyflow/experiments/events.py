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
@dataclass
class EventTable:
    """Flat per-row table of all scored pair-snapshots of one split."""
    scenario: np.ndarray       # (R,) int
    t: np.ndarray              # (R,) int snapshot index within the scenario
    key: np.ndarray            # (R,) int64 unordered pair id
    label: np.ndarray          # (R,) bool
    alert: np.ndarray          # (R,) bool
    ttc: np.ndarray            # (R,) float
    cause: np.ndarray          # (R,) int8
    num_uavs: int
    snapshots_per_scenario: int
    n_scenarios: int

    @staticmethod
    def from_scores(scores: Sequence[SnapshotScores], threshold: float, snapshots_per_scenario: int) -> "EventTable":
        if snapshots_per_scenario <= 0:
            raise ValueError("snapshots_per_scenario must be positive")
        sc, tt, kk, ll, aa, cc, tc = [], [], [], [], [], [], []
        n_uavs = 0
        for s in scores:
            n_uavs = max(n_uavs, int(s.num_uavs))
            P = s.scores.size
            if P == 0:
                continue
            i, j = s.pairs[0], s.pairs[1]
            lo, hi = np.minimum(i, j), np.maximum(i, j)
            sc.append(np.full(P, s.index // snapshots_per_scenario, dtype=np.int64))
            tt.append(np.full(P, s.index % snapshots_per_scenario, dtype=np.int64))
            kk.append(lo.astype(np.int64) * (1 << 20) + hi.astype(np.int64))
            ll.append(s.labels >= 0.5)
            aa.append(s.scores >= threshold)
            cc.append(s.ttc.astype(np.float64))
            tc.append(s.cause.astype(np.int8))
        n_snap = (max(s.index for s in scores) + 1) if scores else 0
        n_scen = int(np.ceil(n_snap / snapshots_per_scenario)) if n_snap else 0
        cat = (lambda xs, dt: np.concatenate(xs) if xs else np.zeros(0, dt))
        return EventTable(cat(sc, np.int64), cat(tt, np.int64), cat(kk, np.int64), cat(ll, bool), cat(aa, bool),
                          cat(cc, np.float64), cat(tc, np.int8), n_uavs, snapshots_per_scenario, n_scen)


def persistent_alerts(tab: EventTable, m: int) -> np.ndarray:
    """M-of-M persistence filter (standard STCA-style post-processing, applied
    identically to every method): a pair is alerted at snapshot t only if its
    score was above threshold at t and at the m-1 immediately preceding
    snapshots (a missing snapshot breaks the run).  m=1 returns ``tab.alert``."""
    if m <= 1:
        return tab.alert
    idx, run = _runs(tab.scenario, tab.key, tab.t, tab.alert)
    out = np.zeros_like(tab.alert)
    if idx.size == 0:
        return out
    first = np.r_[0, np.nonzero(np.diff(run))[0] + 1]
    start_of_run = np.repeat(first, np.diff(np.r_[first, idx.size]))
    position = np.arange(idx.size) - start_of_run          # 0-based position inside its run
    out[idx[position >= m - 1]] = True
    return out


def _runs(scenario: np.ndarray, key: np.ndarray, t: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Group the rows selected by ``mask`` into maximal runs of consecutive
    snapshots (same scenario and pair).  Returns (row indices sorted by run,
    run id per sorted row)."""
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return idx, np.zeros(0, np.int64)
    order = np.lexsort((t[idx], key[idx], scenario[idx]))
    idx = idx[order]
    s, k, tt = scenario[idx], key[idx], t[idx]
    new = np.ones(idx.size, dtype=bool)
    new[1:] = (s[1:] != s[:-1]) | (k[1:] != k[:-1]) | (tt[1:] != tt[:-1] + 1)
    return idx, np.cumsum(new) - 1


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

    def to_row(self) -> Dict[str, float]:
        d = {
            "persistence": self.persistence,
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
        return d


def _stats(x: np.ndarray) -> Dict[str, float]:
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p10": float("nan")}
    return {"mean": float(x.mean()), "median": float(np.median(x)), "p10": float(np.percentile(x, 10))}


def event_metrics(tab: EventTable, lead_s: float = 10.0, snapshot_dt_s: float = 1.0,
                  persistence: int = 1) -> EventResult:
    alert = persistent_alerts(tab, int(persistence))
    # ---- conflict events = runs of positive rows
    pos_idx, ev = _runs(tab.scenario, tab.key, tab.t, tab.label)
    n_events = int(ev.max() + 1) if ev.size else 0
    detected = np.zeros(n_events, dtype=bool)
    first_alert_ttc = np.full(n_events, np.nan)
    max_ttc = np.zeros(n_events)
    cause = np.full(n_events, -1, dtype=np.int64)
    if n_events:
        # rows are sorted by (scenario, key, t) -> first row of each run is its earliest snapshot
        first_row = np.r_[0, np.nonzero(np.diff(ev))[0] + 1]
        max_ttc = tab.ttc[pos_idx[first_row]]
        cause = tab.cause[pos_idx[first_row]].astype(np.int64)
        alerted = alert[pos_idx]
        detected = np.bincount(ev[alerted], minlength=n_events) > 0
        # ttc at the earliest alerted snapshot of each event
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
    al_idx, ep = _runs(tab.scenario, tab.key, tab.t, alert)
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
    )


def evaluate_events(lm, data, snapshots_per_scenario: int, lead_s: float = 10.0,
                    snapshot_dt_s: float = 1.0, threshold: Optional[float] = None,
                    persistence: Sequence[int] = (1,)) -> List[EventResult]:
    """Score the split once (at the method's validation-selected threshold
    unless ``threshold`` is given) and return one EventResult per
    persistence value."""
    thr = float(threshold) if threshold is not None else method_threshold(lm)
    scores = list(iter_scores(lm, data))
    tab = EventTable.from_scores(scores, thr, snapshots_per_scenario)
    return [event_metrics(tab, lead_s=lead_s, snapshot_dt_s=snapshot_dt_s, persistence=m) for m in persistence]
