"""S7a: disk cache round-trip must equal the live build exactly."""

import copy

import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.cache import build_split, cache_key, get_split, load_dataset, save_dataset


def _tiny_cfg():
    cfg = SkyFlowConfig()
    cfg.data.num_uavs = 30
    cfg.data.grid_size_m = 1000.0
    cfg.data.train_scenarios = 1
    cfg.data.val_scenarios = 1
    cfg.data.test_scenarios = 1
    cfg.data.scenario_duration_s = 10.0
    return cfg


def _assert_same(a, b):
    assert len(a) == len(b)
    for (sa, la), (sb, lb) in zip(a, b):
        assert torch.equal(la, lb)
        assert torch.equal(sa.node_features, sb.node_features)
        assert torch.equal(sa.conflict_pairs, sb.conflict_pairs)
        assert torch.equal(sa.conflict_labels, sb.conflict_labels)
        assert torch.equal(sa.conflict_ttc, sb.conflict_ttc)
        assert torch.equal(sa.conflict_cause, sb.conflict_cause)
        assert sa.edge_indices.keys() == sb.edge_indices.keys()
        for r in sa.edge_indices:
            assert torch.equal(sa.edge_indices[r], sb.edge_indices[r])
            assert torch.equal(sa.edge_deltas[r], sb.edge_deltas[r])
        assert sa.num_uavs == sb.num_uavs
        assert sa.num_missed_positives == sb.num_missed_positives
        assert sa.num_pair_candidates == sb.num_pair_candidates


def test_cache_roundtrip_equals_live_build(tmp_path):
    cfg = _tiny_cfg()
    live = build_split(cfg, "val")
    p = tmp_path / "val.pt"
    save_dataset(live, p)
    loaded = load_dataset(p)
    _assert_same(live, loaded)
    # get_split builds + caches, then serves from cache
    built = get_split(cfg, "val", cache_dir=str(tmp_path / "c"), verbose=False)
    again = get_split(cfg, "val", cache_dir=str(tmp_path / "c"), build_if_missing=False, verbose=False)
    _assert_same(built, again)
    _assert_same(live, again)


def test_cache_key_depends_on_data_not_model_seed():
    cfg = _tiny_cfg()
    k0 = cache_key(cfg, "train")
    c2 = copy.deepcopy(cfg)
    c2.training.seed = 999
    c2.model.hidden_dim = 32
    assert cache_key(c2, "train") == k0
    c3 = copy.deepcopy(cfg)
    c3.sim.adsb_latency_s = [0.0, 3.0]
    assert cache_key(c3, "train") != k0
    c4 = copy.deepcopy(cfg)
    c4.features.input_set = "telemetry_only"
    assert cache_key(c4, "train") != k0
    assert cache_key(cfg, "val") != k0


def test_build_split_uses_sim_seed_not_training_seed():
    cfg = _tiny_cfg()
    a = build_split(cfg, "val")
    c2 = copy.deepcopy(cfg)
    c2.training.seed = 7
    b = build_split(c2, "val")
    _assert_same(a, b)
