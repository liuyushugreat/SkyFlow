"""S2: label-leakage guards.

1. Leakage-free feature names exclude d_min / t_cpa / f_avoid.
2. Leakage-free relation vocabulary excludes conflicts_with.
3. Model input dimensions adapt automatically (20 features, 5 relations).
4. Pair edge features e_ij contain only Δp, Δv, δ.
5. Mock future-tamper test: perturbing the truth *after* epoch t leaves the
   graph and features built at t bit-for-bit unchanged (while labels change).
"""

import copy
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import pytest

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import (
    TKGBuilder, LEAKING_FEATURES, LEAKING_RELATIONS,
    uav_feature_names, relation_vocab, LEGACY_RELATION_VOCAB,
)
from skyflow.data.urbanair500 import UrbanAir500
from skyflow.models.tr_gat import TRGAT
from skyflow.models.conflict_head import (
    ConflictScoringHead, build_pair_edge_features, PAIR_EDGE_FEATURE_DIM,
)


class TestFeatureSet:
    def test_leakage_free_feature_names(self):
        names = uav_feature_names(leakage_free=True, plan_context=False)
        assert len(names) == 20
        for leak in LEAKING_FEATURES:
            assert leak not in names
        # S8d default adds the 12-d filed-plan context, still no leaking feature
        full = uav_feature_names(leakage_free=True)
        assert len(full) == 32 and full[:20] == names
        for leak in LEAKING_FEATURES:
            assert leak not in full

    def test_legacy_feature_names_still_available(self):
        names = uav_feature_names(leakage_free=False, plan_context=False)
        assert len(names) == 23
        for leak in LEAKING_FEATURES:
            assert leak in names

    def test_default_builder_is_leakage_free(self):
        b = TKGBuilder()
        assert b.leakage_free is True
        assert b.feature_dim == 32
        assert TKGBuilder(plan_context=False).feature_dim == 20
        assert b.num_relations == 5


class TestRelationSet:
    def test_leakage_free_vocab(self):
        vocab = relation_vocab(leakage_free=True)
        assert len(vocab) == 5
        for leak in LEAKING_RELATIONS:
            assert leak not in vocab
        assert sorted(vocab.values()) == list(range(5))

    def test_legacy_vocab_unchanged(self):
        assert relation_vocab(leakage_free=False) == LEGACY_RELATION_VOCAB
        assert "conflicts_with" in LEGACY_RELATION_VOCAB


class TestDimensionsAdapt:
    def test_config_derived_dims(self):
        cfg = SkyFlowConfig()
        assert cfg.leakage_free() is True
        assert cfg.plan_context() is True
        assert cfg.uav_feature_dim() == 32
        cfg.features.plan_context = False
        assert cfg.uav_feature_dim() == 20
        assert cfg.num_relations() == 5
        cfg.features.leakage_free = False
        assert cfg.uav_feature_dim() == 23
        assert cfg.num_relations() == 6

    def test_model_runs_on_built_snapshot(self):
        cfg = SkyFlowConfig()
        sim = cfg.make_simulator(num_uavs=25, seed=1)
        sim.grid_size = 600.0
        sim._init_infrastructure()
        plans = sim.generate_flight_plans(25)
        log = sim.run_physics(plans, 1.0, extra_seconds=0.0)
        snap = cfg.make_builder().build(sim.observe(log, 5))
        assert snap.node_features.shape[1] == 32
        assert all(k < 5 for k in snap.edge_indices)
        model = TRGAT(node_feature_dim=cfg.uav_feature_dim(),
                      num_relations=cfg.num_relations())
        emb, state = model(snap.node_features, snap.edge_indices, snap.edge_deltas)
        assert emb.shape == (snap.num_nodes, 128)


class TestEdgeFeatures:
    def test_edge_features_are_relative_kinematics_only(self):
        nf = torch.zeros(4, 20)
        nf[:, 0:3] = torch.tensor([[0., 0., 50.], [100., 0., 50.], [0., 300., 60.], [5., 5., 55.]])
        nf[:, 3:6] = torch.tensor([[10., 0., 0.], [-10., 0., 0.], [0., 0., 0.], [1., 1., 0.]])
        pairs = torch.tensor([[0, 0], [1, 2]])
        e = build_pair_edge_features(nf, pairs, uav_aoi=torch.tensor([0.0, 0.5, 2.0, 0.0]))
        assert e.shape == (2, PAIR_EDGE_FEATURE_DIM)
        assert torch.allclose(e[0, 0:3], torch.tensor([1.0, 0.0, 0.0]))     # Δp / 100
        assert torch.allclose(e[0, 3:6], torch.tensor([-2.0, 0.0, 0.0]))    # Δv / 10
        assert e[0, 6].item() == pytest.approx(0.5)                          # max AoI
        assert e[1, 6].item() == pytest.approx(2.0)

    def test_head_accepts_edge_features(self):
        head = ConflictScoringHead()
        P = 6
        out = head(torch.randn(P, 128), torch.randn(P, 128),
                   torch.randn(P, 64), torch.randn(P, 64),
                   edge_feat=torch.randn(P, PAIR_EDGE_FEATURE_DIM))
        assert out.shape == (P,)


class TestFutureTamperInvariance:
    """Graph/features at epoch t must not depend on anything after t."""

    def _make(self):
        sim = UrbanAir500(num_uavs=40, grid_size=800.0, altitude_range=(60.0, 80.0),
                          num_sectors=4, num_weather_cells=4, num_restricted_zones=2,
                          seed=7, label_mode="lookahead", lookahead_s=5.0)
        plans = sim.generate_flight_plans(40)
        log = sim.run_physics(plans, 2.0, extra_seconds=5.0)
        return sim, log

    def _snapshot_dict(self, snap):
        return {
            "x": snap.node_features.clone(),
            "types": snap.node_types.clone(),
            "ei": {k: v.clone() for k, v in snap.edge_indices.items()},
            "ed": {k: v.clone() for k, v in snap.edge_deltas.items()},
        }

    def test_tampering_future_leaves_graph_unchanged(self):
        sim, log = self._make()
        t_epoch = 10

        builder_a = TKGBuilder()
        for e in range(0, t_epoch + 1):           # build history so δ cache is populated
            snap_a = builder_a.build(sim.observe(log, e))
        ref = self._snapshot_dict(snap_a)
        labels_ref = {(c.uav_i, c.uav_j) for c in sim.label(log, t_epoch)}

        # Tamper: collapse every future position onto a single point so that
        # *all* pairs become future conflicts.
        log_t = copy.deepcopy(log)
        log_t.positions[t_epoch + 1:] = np.array([400.0, 400.0, 70.0], dtype=np.float32)
        log_t.velocities[t_epoch + 1:] += 3.0

        sim_t = copy.deepcopy(sim)
        builder_b = TKGBuilder()
        for e in range(0, t_epoch + 1):
            snap_b = builder_b.build(sim_t.observe(log_t, e))
        got = self._snapshot_dict(snap_b)
        labels_t = {(c.uav_i, c.uav_j) for c in sim_t.label(log_t, t_epoch)}

        assert torch.equal(ref["x"], got["x"])
        assert torch.equal(ref["types"], got["types"])
        assert ref["ei"].keys() == got["ei"].keys()
        for k in ref["ei"]:
            assert torch.equal(ref["ei"][k], got["ei"][k])
            assert torch.equal(ref["ed"][k], got["ed"][k])
        # sanity: the tamper did change the ground truth (many more positives)
        assert labels_t != labels_ref
        assert len(labels_t) > len(labels_ref) + 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
