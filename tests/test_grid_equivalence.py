"""S4: grid-hash neighbour search must reproduce the brute-force graph exactly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from skyflow.data.tkg_builder import TKGBuilder, CANDIDATE_RELATIONS
from skyflow.data.urbanair500 import UrbanAir500


def _random_state(n_uavs, grid_size, seed, epoch=50):
    sim = UrbanAir500(num_uavs=n_uavs, grid_size=grid_size, seed=seed,
                      num_sectors=16, num_weather_cells=16, num_restricted_zones=4)
    plans = sim.generate_flight_plans(n_uavs)
    log = sim.run_physics(plans, (epoch + 1) * sim.dt, extra_seconds=0.0)
    return sim.observe(log, epoch)


def _edge_set(snap, rel_idx):
    if rel_idx not in snap.edge_indices:
        return set()
    ei = snap.edge_indices[rel_idx].numpy()
    return set(zip(ei[0].tolist(), ei[1].tolist()))


@pytest.mark.parametrize("n_uavs,grid_size", [(100, 2000.0), (500, 5000.0), (1000, 8000.0)])
def test_grid_equals_bruteforce(n_uavs, grid_size):
    state = _random_state(n_uavs, grid_size, seed=n_uavs)
    grid = TKGBuilder(neighbor_search="grid")
    brute = TKGBuilder(neighbor_search="bruteforce")
    sg = grid.build(state)
    sb = brute.build(state)

    # approaches edges: identical sets and identical ordering/deltas
    a = grid.relations["approaches"]
    assert _edge_set(sg, a) == _edge_set(sb, a)
    assert len(_edge_set(sb, a)) > 0, "scenario produced no approaches edges"
    if a in sg.edge_indices:
        assert torch.equal(sg.edge_indices[a], sb.edge_indices[a])
        assert torch.equal(sg.edge_deltas[a], sb.edge_deltas[a])

    # all other relations untouched
    for r, idx in grid.relations.items():
        assert _edge_set(sg, idx) == _edge_set(sb, idx), r

    # node features and candidate pairs identical
    assert torch.equal(sg.node_features, sb.node_features)
    assert torch.equal(sg.candidate_pairs, sb.candidate_pairs)

    # the grid really pruned something and the pruning is a subset
    assert sg.num_pair_candidates <= sb.num_pair_candidates == n_uavs * (n_uavs - 1) // 2


def test_candidate_pairs_cover_all_candidate_relation_edges():
    state = _random_state(300, 3000.0, seed=7)
    b = TKGBuilder(neighbor_search="grid")
    snap = b.build(state)
    cand = set(map(tuple, snap.candidate_pairs.T.tolist()))
    for r in CANDIDATE_RELATIONS:
        for i, j in _edge_set(snap, b.relations[r]):
            assert (min(i, j), max(i, j)) in cand
    # no duplicates, i < j
    assert len(cand) == snap.candidate_pairs.shape[1]
    assert bool((snap.candidate_pairs[0] < snap.candidate_pairs[1]).all())


def test_grid_candidates_contain_all_close_pairs():
    """Every pair with horizontal distance < cell must be a grid candidate."""
    rng = np.random.RandomState(0)
    P = rng.uniform(0, 3000, size=(400, 3)).astype(np.float32)
    cell = 350.0
    ci, cj = TKGBuilder._grid_candidates(P, cell)
    cand = set(zip(ci.tolist(), cj.tolist()))
    ii, jj = np.triu_indices(400, k=1)
    d = np.hypot(P[jj, 0] - P[ii, 0], P[jj, 1] - P[ii, 1])
    for i, j in zip(ii[d < cell].tolist(), jj[d < cell].tolist()):
        assert (i, j) in cand
    # sorted lexicographically, unique
    assert len(cand) == len(ci)
    assert np.all(np.lexsort((cj, ci)) == np.arange(len(ci)))


def test_candidate_radius_is_conservative():
    b = TKGBuilder(approach_cpa_h=80.0, approach_lookahead=60.0)
    V = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0]], np.float32)   # |v|max = 5
    assert b.candidate_radius(V) == pytest.approx(80.0 + 2 * 5.0 * 60.0)


def test_dataset_edges_mode_records_missed_positives():
    sim = UrbanAir500(num_uavs=60, grid_size=800.0, seed=3, lookahead_s=30.0,
                      num_sectors=4, num_weather_cells=4, num_restricted_zones=1)
    b = TKGBuilder(neighbor_search="grid")
    data_edges = sim.generate_dataset("val", 1, 5.0, builder=b, candidates="edges")
    data_all = sim.generate_dataset("val", 1, 5.0, builder=b, candidates="all")
    n_pos_edges = sum(int(s.conflict_labels.sum()) + s.num_missed_positives for s, _ in data_edges)
    n_pos_all = sum(int(s.conflict_labels.sum()) for s, _ in data_all)
    assert n_pos_all > 0
    assert n_pos_edges == n_pos_all               # positives are conserved
    for s, _ in data_all:
        assert s.num_missed_positives == 0
        assert s.conflict_pairs.shape[1] == 60 * 59 // 2
    for s, _ in data_edges:
        assert torch.equal(s.conflict_pairs, s.candidate_pairs)
        assert s.missed_ttc.numel() == s.num_missed_positives


@pytest.mark.parametrize("n_uavs,grid_size,seed", [(200, 1500.0, 1), (120, 1000.0, 5), (300, 5000.0, 9)])
def test_proximity_candidates_have_full_recall(n_uavs, grid_size, seed):
    sim = UrbanAir500(num_uavs=n_uavs, grid_size=grid_size, seed=seed, lookahead_s=30.0,
                      num_sectors=4, num_weather_cells=4, num_restricted_zones=1)
    b = TKGBuilder(neighbor_search="grid")
    data = sim.generate_dataset("val", 1, 20.0, builder=b, candidates="proximity")
    n_pos = sum(int(s.conflict_labels.sum()) for s, _ in data)
    n_missed = sum(s.num_missed_positives for s, _ in data)
    assert n_pos > 0
    assert n_missed == 0                          # recall 1 by construction
    n_all = n_uavs * (n_uavs - 1) // 2
    n_cand = np.mean([s.conflict_pairs.shape[1] for s, _ in data])
    if grid_size >= 5000.0:
        assert n_cand < 0.8 * n_all               # prunes on the paper's 5 km area
    for s, _ in data:
        assert bool((s.conflict_pairs[0] < s.conflict_pairs[1]).all())
        assert s.conflict_pairs.shape[1] == len(set(map(tuple, s.conflict_pairs.T.tolist())))


def test_proximity_with_delayed_observations_keeps_recall():
    sim = UrbanAir500(num_uavs=200, grid_size=1500.0, seed=2, lookahead_s=30.0,
                      adsb_latency_range=(1.0, 2.0), packet_loss=0.2,
                      num_sectors=4, num_weather_cells=4, num_restricted_zones=1)
    data = sim.generate_dataset("val", 1, 20.0, candidates="proximity")
    assert sum(int(s.conflict_labels.sum()) for s, _ in data) > 0
    assert sum(s.num_missed_positives for s, _ in data) == 0


def test_invalid_modes_rejected():
    with pytest.raises(ValueError):
        TKGBuilder(neighbor_search="kd_tree")
    sim = UrbanAir500(num_uavs=5, seed=0)
    with pytest.raises(ValueError):
        sim.generate_dataset("val", 1, 1.0, candidates="random")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
