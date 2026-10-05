"""S5: CPA-Rule baseline."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from skyflow.baselines.cpa_rule import CPARule, conflict_within_window
from skyflow.baselines.registry import REGISTRY, get_baseline, is_deterministic
from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGSnapshot


def _snapshot(positions, velocities, pairs):
    n = len(positions)
    feats = torch.zeros(n, 20)
    feats[:, 0:3] = torch.tensor(positions, dtype=torch.float32)
    feats[:, 3:6] = torch.tensor(velocities, dtype=torch.float32)
    return TKGSnapshot(
        node_features=feats, node_types=torch.zeros(n, dtype=torch.long),
        edge_indices={}, edge_deltas={}, num_uavs=n, num_nodes=n,
        conflict_pairs=torch.tensor(pairs, dtype=torch.long).T,
    )


def test_head_on_pair_is_flagged():
    snap = _snapshot([[0, 0, 100], [200, 0, 100]], [[10, 0, 0], [-10, 0, 0]], [(0, 1)])
    assert CPARule(window_s=30.0).predict(snap).tolist() == [1.0]


def test_parallel_diverging_pair_is_not_flagged():
    snap = _snapshot([[0, 0, 100], [0, 50, 100]], [[10, 0, 0], [10, 1, 0]], [(0, 1)])
    assert CPARule(window_s=30.0).predict(snap).tolist() == [0.0]


def test_vertical_separation_prevents_conflict():
    snap = _snapshot([[0, 0, 100], [200, 0, 130]], [[10, 0, 0], [-10, 0, 0]], [(0, 1)])
    assert CPARule(window_s=30.0, v_thresh=3.0).predict(snap).tolist() == [0.0]


def test_conflict_beyond_window_is_not_flagged():
    # closing at 10 m/s relative, 1000 m apart -> meet at t = 100 s > 30 s
    snap = _snapshot([[0, 0, 100], [1000, 0, 100]], [[5, 0, 0], [-5, 0, 0]], [(0, 1)])
    assert CPARule(window_s=30.0).predict(snap).tolist() == [0.0]
    assert CPARule(window_s=120.0).predict(snap).tolist() == [1.0]


def test_simultaneity_matters():
    # Horizontal pass at t=10 s, but vertical crossing only at t=25 s: never both inside
    snap = _snapshot([[0, 0, 100], [200, 0, 100 + 50]], [[10, 0, 0], [-10, 0, -2]], [(0, 1)])
    # vertical |50 - 2t| < 3  <=> t in (23.5, 26.5); horizontal |200-20t| < 10 <=> t in (9.5, 10.5)
    assert CPARule(window_s=30.0).predict(snap).tolist() == [0.0]


def test_vectorised_test_matches_brute_force_sampling():
    rng = np.random.RandomState(0)
    dp = rng.uniform(-300, 300, size=(2000, 3)); dp[:, 2] *= 0.1
    dv = rng.uniform(-20, 20, size=(2000, 3)); dv[:, 2] *= 0.1
    W, h, v = 30.0, 10.0, 3.0
    exact = conflict_within_window(dp, dv, W, h, v)
    ts = np.linspace(0, W, 30001)
    rel = dp[:, None, :] + dv[:, None, :] * ts[None, :, None]
    sampled = ((np.hypot(rel[..., 0], rel[..., 1]) < h) & (np.abs(rel[..., 2]) < v)).any(axis=1)
    # dense sampling can only miss razor-thin intervals -> exact ⊇ sampled; agreement should be near-perfect
    assert np.all(exact[sampled])
    assert np.mean(exact == sampled) > 0.995


def test_val_search_recovers_label_thresholds_on_consistent_data():
    rng = np.random.RandomState(1)
    snaps = []
    for _ in range(20):
        n = 30
        P = rng.uniform(0, 600, size=(n, 3)); P[:, 2] = 100 + rng.uniform(-6, 6, n)
        V = rng.uniform(-12, 12, size=(n, 3)); V[:, 2] = 0.0
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        snap = _snapshot(P, V, pairs)
        dp = P[[j for _, j in pairs]] - P[[i for i, _ in pairs]]
        dv = V[[j for _, j in pairs]] - V[[i for i, _ in pairs]]
        labels = torch.tensor(conflict_within_window(dp, dv, 30.0, 10.0, 3.0).astype(np.float32))
        snaps.append((snap, labels))
    rule = CPARule(window_s=30.0, threshold_mode="val_search")
    h, v = rule.fit(snaps)
    assert (h, v) == (10.0, 3.0)
    assert rule.search_log and max(r["f1"] for r in rule.search_log) == pytest.approx(1.0)


def test_label_mode_ignores_fit():
    rule = CPARule(threshold_mode="label")
    assert rule.fit([]) == (10.0, 3.0)
    with pytest.raises(ValueError):
        CPARule(threshold_mode="auto")


def test_registry_builds_cpa_rule_from_config():
    cfg = SkyFlowConfig()
    assert "CPA-Rule" in REGISTRY and is_deterministic("CPA-Rule")
    rule = get_baseline("CPA-Rule", cfg, torch.device("cpu"))
    assert isinstance(rule, CPARule)
    assert rule.window_s == cfg.data.lookahead_seconds
    assert rule.threshold_mode == cfg.baselines.cpa_rule_thresholds
    assert rule.count_parameters() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
