"""S8c: shared pair feature e_ij with observed-state CPA geometry.

Checks that
  * the geometry features are the plain linear-extrapolation CPA quantities of
    the *observed* relative state (same inputs as CPA-Rule, no labels),
  * the three ``features.pair_edge_features`` modes give the documented widths,
  * every learned scorer (TR-GAT head and the four learned baselines) receives
    e_ij and runs in all three modes, and ``none`` reproduces the legacy
    [h_i, h_j] scorer width,
  * the cache signature ignores the switch (model-side feature).
"""

from __future__ import annotations

import math

import pytest
import torch

from skyflow.config import SkyFlowConfig
from skyflow.baselines.gat_static import GATStatic
from skyflow.baselines.lstm_pair import LSTMPair
from skyflow.baselines.registry import REGISTRY
from skyflow.baselines.stgcn import STGCN
from skyflow.baselines.transformer_pair import TransformerPair
from skyflow.baselines.cpa_rule import conflict_within_window
from skyflow.data.cache import data_signature
from skyflow.data.tkg_builder import TKGSnapshot
from skyflow.models.conflict_head import (
    PAIR_EDGE_FEATURE_DIM,
    PAIR_GEOMETRY_DIM,
    build_pair_edge_features,
    observed_cpa_geometry,
    pair_edge_feature_dim,
)


def _two_uavs():
    """UAV 0 at origin flying +x at 10 m/s; UAV 1 at (300, 20, 55) flying -x at 10 m/s."""
    nf = torch.zeros(2, 20)
    nf[0, 0:3] = torch.tensor([0.0, 0.0, 50.0])
    nf[1, 0:3] = torch.tensor([300.0, 20.0, 55.0])
    nf[0, 3:6] = torch.tensor([10.0, 0.0, 0.0])
    nf[1, 3:6] = torch.tensor([-10.0, 0.0, 0.0])
    return nf


class TestGeometry:
    def test_head_on_pair_matches_hand_computation(self):
        nf = _two_uavs()
        pairs = torch.tensor([[0], [1]])
        g = observed_cpa_geometry(nf[pairs[1], 0:3] - nf[pairs[0], 0:3],
                                  nf[pairs[1], 3:6] - nf[pairs[0], 3:6], window_s=30.0)
        assert g.shape == (1, PAIR_GEOMETRY_DIM)
        # relative velocity -20 m/s along x, relative position 300 m -> t_cpa = 15 s
        assert g[0, 0].item() == pytest.approx(15.0 / 30.0)
        # at t_cpa the x-offset is 0, lateral offset stays 20 m -> d_cpa_h = 20 m
        assert g[0, 1].item() == pytest.approx(20.0 / 100.0)
        # no vertical velocity -> |dz| = 5 m
        assert g[0, 2].item() == pytest.approx(5.0 / 10.0)
        # current horizontal range sqrt(300^2 + 20^2)
        assert g[0, 3].item() == pytest.approx(math.hypot(300.0, 20.0) / 100.0)
        # closing speed = -(dp.dv)/|dp| = 20*300/300.67
        assert g[0, 4].item() == pytest.approx((20.0 * 300.0 / math.hypot(300.0, 20.0)) / 10.0, rel=1e-5)

    def test_t_cpa_is_clipped_to_window_and_zero_when_diverging(self):
        dp = torch.tensor([[1000.0, 0.0, 0.0], [100.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
        dv = torch.tensor([[-1.0, 0.0, 0.0],      # converging slowly -> t_cpa 1000 s, clipped to 30
                           [5.0, 0.0, 0.0],        # diverging -> t_cpa 0
                           [0.0, 0.0, 0.0]])       # parallel -> t_cpa 0, d_cpa = range
        g = observed_cpa_geometry(dp, dv, window_s=30.0)
        assert g[0, 0].item() == pytest.approx(1.0)
        assert g[1, 0].item() == pytest.approx(0.0)
        assert g[2, 0].item() == pytest.approx(0.0)
        assert g[2, 1].item() == pytest.approx(1.0)
        assert torch.isfinite(g).all()

    def test_geometry_separates_cpa_rule_decisions(self):
        """Pairs the CPA rule flags must have d_cpa_h < 10 m and |dz_cpa| < 3 m
        (necessary condition of the rule) - the learned head therefore has the
        rule's decision variables as inputs, computed from observed state only."""
        torch.manual_seed(0)
        P = 2000
        dp = torch.randn(P, 3) * torch.tensor([200.0, 200.0, 10.0])
        dv = torch.randn(P, 3) * torch.tensor([8.0, 8.0, 1.0])
        hit = conflict_within_window(dp.double().numpy(), dv.double().numpy(), 30.0, 10.0, 3.0)
        g = observed_cpa_geometry(dp, dv, 30.0)
        hit_t = torch.from_numpy(hit)
        assert hit_t.any(), "test needs some positives"
        assert (g[hit_t, 1] * 100.0 < 10.0 + 1e-3).all()


class TestModes:
    def test_dims(self):
        assert pair_edge_feature_dim("none") == 0
        assert pair_edge_feature_dim("kinematics") == PAIR_EDGE_FEATURE_DIM == 7
        assert pair_edge_feature_dim("geometry") == PAIR_EDGE_FEATURE_DIM + PAIR_GEOMETRY_DIM == 12
        with pytest.raises(ValueError):
            pair_edge_feature_dim("bogus")

    def test_build_modes(self):
        nf = _two_uavs()
        pairs = torch.tensor([[0], [1]])
        aoi = torch.tensor([0.0, 1.5])
        assert build_pair_edge_features(nf, pairs, aoi, mode="none") is None
        k = build_pair_edge_features(nf, pairs, aoi, mode="kinematics")
        g = build_pair_edge_features(nf, pairs, aoi, mode="geometry")
        assert k.shape == (1, 7) and g.shape == (1, 12)
        assert torch.allclose(g[:, :7], k)          # geometry is a superset of kinematics
        assert k[0, 6].item() == pytest.approx(1.5)

    def test_default_config_is_geometry_and_legacy_modes_exist(self):
        cfg = SkyFlowConfig()
        assert cfg.features.pair_edge_features == "geometry"
        cfg.features.pair_edge_features = "none"
        assert pair_edge_feature_dim(cfg.features.pair_edge_features) == 0


def _snapshot(n_uav=6, n_pairs=5):
    torch.manual_seed(1)
    nf = torch.randn(n_uav, 20)
    nf[:, 0:3] *= 200.0
    nf[:, 3:6] *= 8.0
    pairs = torch.stack([torch.arange(n_pairs), (torch.arange(n_pairs) + 1) % n_uav])
    src = torch.arange(n_uav)
    dst = (src + 1) % n_uav
    return TKGSnapshot(
        node_features=nf,
        node_types=torch.zeros(n_uav, dtype=torch.long),
        edge_indices={0: torch.stack([src, dst])},
        edge_deltas={0: torch.zeros(n_uav)},
        num_uavs=n_uav,
        num_nodes=n_uav,
        conflict_pairs=pairs,
        uav_aoi=torch.rand(n_uav),
    )


class TestLearnedScorersReceivePairFeatures:
    @pytest.mark.parametrize("cls", [GATStatic, STGCN, LSTMPair, TransformerPair])
    @pytest.mark.parametrize("mode", ["none", "kinematics", "geometry"])
    def test_forward_all_modes(self, cls, mode):
        m = cls(input_dim=20, pair_edge_features=mode)
        m.eval()
        snap = _snapshot()
        out = m(snap)
        assert out.shape == (snap.conflict_pairs.size(1),)
        first = m.scorer[0]
        base = first.in_features - pair_edge_feature_dim(mode)
        assert base > 0 and base % 2 == 0           # [h_i, h_j] + e_ij

    def test_none_reproduces_legacy_width(self):
        legacy = GATStatic(input_dim=20, pair_edge_features="none")
        assert legacy.scorer[0].in_features == 2 * 128
        new = GATStatic(input_dim=20)
        assert new.scorer[0].in_features == 2 * 128 + 12

    def test_registry_passes_config(self):
        cfg = SkyFlowConfig()
        cfg.features.pair_edge_features = "kinematics"
        for name in ("LSTM-P", "Tfm-P", "STGCN", "GAT-S"):
            m = REGISTRY[name].factory(cfg, torch.device("cpu"))
            assert m.pair_edge_features == "kinematics"
            assert m.window_s == cfg.data.lookahead_seconds

    def test_trgat_head_width_follows_config(self):
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = SkyFlowConfig()
        for mode, dim in (("none", 0), ("kinematics", 7), ("geometry", 12)):
            cfg.features.pair_edge_features = mode
            tr = SkyFlowTrainer(cfg, device=torch.device("cpu"))
            tr.build_model()
            assert tr.head.edge_feature_dim == dim


class TestCacheSignature:
    def test_switch_does_not_change_cache_key(self):
        cfg = SkyFlowConfig()
        a = data_signature(cfg, "train")
        cfg.features.pair_edge_features = "none"
        assert data_signature(cfg, "train") == a
