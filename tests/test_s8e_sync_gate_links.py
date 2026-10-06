"""S8e: AoI-synchronised pair geometry, intent-conformance gate, heterogeneous
links and the unified training schedule.

  * ``geometry_plan_sync`` dead-reckons every report to the common epoch with
    its own age before the pair geometry is computed (hand-checked), carries
    both ages, and reduces to ``geometry_plan`` when all ages are zero;
  * both CPA rules apply the same synchronisation when ``aoi_sync`` is set,
    and the registry derives the flag from the pair mode;
  * the conformance gate is 0 for UAVs without a plan, in (0,1) otherwise,
    w=1 reproduces the planned geometry and w=0 the linear extrapolation;
    the TR-GAT head widens by 9 and trains end-to-end;
  * ``sim.link_mix`` draws deterministic per-scenario link conditions inside
    the configured ranges, explicit obs_params still override, and the cache
    key changes with it;
  * ``training.scheduler`` / ``tbptt_detach`` switches exist and the legacy
    settings are reproducible.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from skyflow.baselines.cpa_rule import CPARule, PlanCPARule
from skyflow.baselines.registry import REGISTRY
from skyflow.config import SkyFlowConfig
from skyflow.data.cache import data_signature
from skyflow.data.tkg_builder import TKGBuilder, TKGSnapshot, uav_feature_names
from skyflow.data.urbanair500 import ObservationParams, UrbanAir500
from skyflow.models.conflict_head import (
    PAIR_GATE_DIM, ConflictScoringHead, ConformanceGate, build_pair_edge_features, conformance_residuals,
    gated_pair_geometry, mode_uses_sync, pair_edge_feature_dim, plan_column, planned_pair_geometry,
    synchronised_positions,
)


def _snap(pos, vel, plan_ctx, aoi):
    n = pos.shape[0]
    nf = torch.zeros(n, 32)
    nf[:, 0:3] = torch.tensor(pos, dtype=torch.float32)
    nf[:, 3:6] = torch.tensor(vel, dtype=torch.float32)
    nf[:, 20:32] = torch.tensor(plan_ctx, dtype=torch.float32)
    return TKGSnapshot(
        node_features=nf, node_types=torch.zeros(n, dtype=torch.long),
        edge_indices={}, edge_deltas={}, num_uavs=n, num_nodes=n,
        conflict_pairs=torch.tensor([[0], [1]]),
        uav_aoi=torch.tensor(aoi, dtype=torch.float32),
        feature_names=uav_feature_names(True, True),
    )


def _head_on(aoi0, aoi1):
    """UAV0 at origin flying +x at 10 m/s, UAV1 at x=300 flying -x at 10 m/s; both plans straight."""
    pos = np.array([[0.0, 0.0, 50.0], [300.0, 0.0, 50.0]])
    vel = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
    ctx = np.zeros((2, 12))
    for k in range(3):
        ctx[0, 3 + 3 * k: 6 + 3 * k] = [100.0 * (k + 1), 0.0, 0.0]
        ctx[1, 3 + 3 * k: 6 + 3 * k] = [-100.0 * (k + 1), 0.0, 0.0]
    return _snap(pos, vel, ctx, [aoi0, aoi1])


class TestSync:
    def test_dead_reckoning_shifts_each_report_by_its_own_age(self):
        s = _head_on(1.0, 0.5)
        p = synchronised_positions(s.node_features, s.uav_aoi, True)
        assert p[0].tolist() == pytest.approx([10.0, 0.0, 50.0])
        assert p[1].tolist() == pytest.approx([295.0, 0.0, 50.0])
        assert torch.equal(synchronised_positions(s.node_features, s.uav_aoi, False), s.node_features[:, 0:3])

    def test_sync_mode_geometry_and_ages(self):
        s = _head_on(1.0, 0.5)
        c0 = plan_column(s.feature_names)
        e = build_pair_edge_features(s.node_features, s.conflict_pairs, s.uav_aoi, mode="geometry_plan_sync",
                                     plan_col0=c0)
        assert e.shape == (1, 22)
        assert e[0, 0].item() == pytest.approx(2.85)            # synchronised dx = 285 m
        assert e[0, 6].item() == pytest.approx(1.0)             # delta = max age
        assert e[0, 20].item() == pytest.approx(1.0) and e[0, 21].item() == pytest.approx(0.5)
        # planned separation at +10 s on synchronised positions: 285 - 200 = 85 m
        assert e[0, 12].item() == pytest.approx(0.85)
        # with zero ages the sync mode equals geometry_plan plus two zero columns
        z = _head_on(0.0, 0.0)
        a = build_pair_edge_features(z.node_features, z.conflict_pairs, z.uav_aoi, mode="geometry_plan_sync", plan_col0=c0)
        b = build_pair_edge_features(z.node_features, z.conflict_pairs, z.uav_aoi, mode="geometry_plan", plan_col0=c0)
        assert torch.allclose(a[:, :20], b) and float(a[:, 20:].abs().sum()) == 0.0
        assert mode_uses_sync("geometry_plan_sync") and not mode_uses_sync("geometry_plan")

    def test_rules_apply_the_same_synchronisation(self):
        # Crossing pair: UAV0 flies +x from the origin, UAV1 flies -y and its report (150, 150) is 3 s old
        # (non-cooperative). Raw reports meet exactly at t = 15 s (hit); once UAV1 is dead-reckoned to
        # (150, 120) the closest approach is 21 m (no hit).
        pos = np.array([[0.0, 0.0, 50.0], [150.0, 150.0, 50.0]])
        vel = np.array([[10.0, 0.0, 0.0], [0.0, -10.0, 0.0]])
        ctx = np.zeros((2, 12))
        s = _snap(pos, vel, ctx, [0.0, 3.0])
        raw = CPARule(window_s=30.0, h_thresh=10.0, v_thresh=3.0, aoi_sync=False)
        syn = CPARule(window_s=30.0, h_thresh=10.0, v_thresh=3.0, aoi_sync=True)
        assert raw.predict(s).tolist() == [1.0] and syn.predict(s).tolist() == [0.0]
        cfg = SkyFlowConfig()
        assert REGISTRY["CPA-Rule"].factory(cfg, torch.device("cpu")).aoi_sync is True
        assert REGISTRY["Plan-CPA"].factory(cfg, torch.device("cpu")).aoi_sync is True
        cfg.features.pair_edge_features = "geometry_plan"
        assert REGISTRY["CPA-Rule"].factory(cfg, torch.device("cpu")).aoi_sync is False
        assert isinstance(REGISTRY["Plan-CPA"].factory(cfg, torch.device("cpu")), PlanCPARule)


class TestConformanceGate:
    def test_residuals_and_gate_range(self):
        s = _head_on(0.0, 0.0)
        c0 = plan_column(s.feature_names)
        r = conformance_residuals(s.node_features, c0)
        assert r.shape == (2, 4)
        assert r[0, 0].item() == pytest.approx(1.0) and r[0, 1].item() == pytest.approx(0.0)
        # UAV without plan: residuals zero, flag 1, gate forced to 0
        s.node_features[1, 20:32] = 0.0
        r = conformance_residuals(s.node_features, c0)
        assert r[1].tolist() == pytest.approx([0.0, 0.0, 0.0, 1.0])
        gate = ConformanceGate(recurrent_dim=8)
        w = gate(torch.zeros(2, 8), s.node_features, c0)
        assert 0.0 < w[0].item() < 1.0 and w[1].item() == 0.0

    def test_gate_extremes_reproduce_plan_and_extrapolation(self):
        # UAV1's plan turns away (+y) while its velocity points -x
        pos = np.array([[0.0, 0.0, 50.0], [300.0, 0.0, 50.0]])
        vel = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        ctx = np.zeros((2, 12))
        for k in range(3):
            ctx[0, 3 + 3 * k: 6 + 3 * k] = [100.0 * (k + 1), 0.0, 0.0]
            ctx[1, 3 + 3 * k: 6 + 3 * k] = [0.0, 100.0 * (k + 1), 0.0]
        s = _snap(pos, vel, ctx, [0.0, 0.0])
        c0 = plan_column(s.feature_names)
        plan = planned_pair_geometry(s.node_features, s.conflict_pairs, c0, with_flag=False)
        g1 = gated_pair_geometry(s.node_features, s.conflict_pairs, c0, torch.ones(2), None, False)
        assert torch.allclose(g1[:, :7], plan) and g1[0, 7].item() == 1.0
        g0 = gated_pair_geometry(s.node_features, s.conflict_pairs, c0, torch.zeros(2), None, False)
        # w = 0: linear extrapolation -> at +10 s separation 300-200 = 100 m, +20 s 100 m (crossed), +30 s 300 m
        assert g0[0, 0].item() == pytest.approx(1.0) and g0[0, 4].item() == pytest.approx(3.0)
        # w = 1: plan -> at +10 s UAV1 is at (300, 100): hypot(200, 100)
        assert g1[0, 0].item() == pytest.approx(np.hypot(200.0, 100.0) / 100.0)

    def test_head_width_and_backward(self):
        s = _head_on(1.0, 0.5)
        c0 = plan_column(s.feature_names)
        head = ConflictScoringHead(embed_dim=16, recurrent_dim=8, edge_feature_dim=22, conformance_gate=True)
        assert head.net[0].in_features == 2 * 16 + 2 * 8 + 22 + PAIR_GATE_DIM
        h = torch.randn(2, 16, requires_grad=True)
        st = torch.randn(2, 8, requires_grad=True)
        e = build_pair_edge_features(s.node_features, s.conflict_pairs, s.uav_aoi, "geometry_plan_sync", 30.0, c0)
        ctx = {"node_features": s.node_features, "pairs": s.conflict_pairs, "plan_col0": c0,
               "uav_aoi": s.uav_aoi, "rec_state": st, "sync": True}
        p = head(h[[0]], h[[1]], st[[0]], st[[1]], e, gate_ctx=ctx)
        p.sum().backward()
        assert st.grad is not None and head.last_gate.shape == (2,)
        with pytest.raises(ValueError):
            head(h[[0]], h[[1]], st[[0]], st[[1]], e, gate_ctx=None)

    def test_trainer_builds_gate_from_config_and_ablation_disables_it(self):
        from skyflow.experiments.methods import method_config
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = SkyFlowConfig()
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
        assert tr.head.conformance_gate is True and tr.head.edge_feature_dim == 22
        for name in ("abl_no_conf_gate", "abl_no_plan"):
            c = method_config(name, SkyFlowConfig())
            tr = SkyFlowTrainer(c, device=torch.device("cpu")); tr.build_model()
            assert tr.head.conformance_gate is False
        assert method_config("abl_no_sync", SkyFlowConfig()).features.pair_edge_features == "geometry_plan"
        assert method_config("abl_tbptt", SkyFlowConfig()).training.tbptt_detach is True

    def test_trgat_end_to_end_step_with_gate_and_bptt(self):
        """One optimisation window on a tiny simulated scene (CPU)."""
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = SkyFlowConfig()
        cfg.model.num_layers, cfg.model.embed_dim, cfg.model.num_heads = 1, 16, 2
        cfg.model.temporal_dim, cfg.model.recurrent_dim = 8, 8
        cfg.data.num_uavs, cfg.data.grid_size_m, cfg.data.observation_window = 20, 600.0, 3
        cfg.data.num_sectors, cfg.data.num_weather_cells, cfg.data.num_restricted_zones = 4, 4, 2
        cfg.training.epochs, cfg.training.min_epochs, cfg.training.early_stopping_patience = 1, 1, 0
        sim = cfg.make_simulator(seed=3)
        data = sim.generate_dataset("train", 1, 6.0, builder=cfg.make_builder())
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            info = tr.train(data, data, seed=1, output_dir=d, max_epochs=1)
        assert info["epochs_run"] == 1 and tr.head.last_gate is not None


class TestLinkMix:
    def test_per_scenario_params_are_deterministic_and_in_range(self):
        cfg = SkyFlowConfig()
        cfg.data.num_uavs, cfg.data.grid_size_m = 10, 400.0
        sim = cfg.make_simulator(seed=11)
        assert sim.link_mix is not None
        logs = sim.simulate_logs("train", 3, 2.0)
        ps = [sim.scenario_obs_params(lg) for lg in logs]
        for p in ps:
            assert 0.0 <= p.packet_loss <= 0.3
            lo, hi = p.latency_range()
            assert 0.3 <= lo <= 0.8 and 1.0 <= hi <= 3.0 and lo <= hi
        assert len({(p.packet_loss, p.latency_range()) for p in ps}) > 1
        again = cfg.make_simulator(seed=11)
        assert [again.scenario_obs_params(lg) for lg in logs] == ps
        # a val scenario with the same index gets a different draw
        logs_val = again.simulate_logs("val", 1, 2.0)
        assert again.scenario_obs_params(logs_val[0]) != ps[0]

    def test_explicit_params_override_and_null_disables(self):
        cfg = SkyFlowConfig()
        cfg.data.num_uavs, cfg.data.grid_size_m = 10, 400.0
        sim = cfg.make_simulator(seed=11)
        logs = sim.simulate_logs("test", 1, 2.0)
        fixed = ObservationParams(adsb_latency_s=(2.0, 2.0), packet_loss=0.25)
        a = sim.dataset_from_logs(logs, obs_params=fixed, builder=cfg.make_builder())
        b = sim.dataset_from_logs(logs, obs_params=fixed, builder=cfg.make_builder())
        assert torch.equal(a[0][0].node_features, b[0][0].node_features)
        cfg.sim.link_mix = None
        plain = cfg.make_simulator(seed=11)
        assert plain.link_mix is None
        assert plain.scenario_obs_params(logs[0]) == plain.obs_params
        with pytest.raises(ValueError):
            UrbanAir500(num_uavs=5, link_mix={"packet_loss": [0.3, 0.1]})
        with pytest.raises(ValueError):
            UrbanAir500(num_uavs=5, link_mix={"bogus": [0.0, 1.0]})

    def test_link_mix_changes_cache_key_but_pair_mode_and_gate_do_not(self):
        cfg = SkyFlowConfig()
        sig = data_signature(cfg, "train")
        cfg.features.pair_edge_features = "geometry_plan"
        cfg.model.use_conformance_gate = False
        cfg.training.tbptt_detach = True
        assert data_signature(cfg, "train") == sig
        cfg.sim.link_mix = None
        assert data_signature(cfg, "train") != sig


class TestSchedule:
    def test_scheduler_switch(self):
        import torch.nn as nn
        from skyflow.training.trainer import build_scheduler
        lin = nn.Linear(2, 1)
        opt = torch.optim.AdamW(lin.parameters(), lr=1e-3)
        sch = build_scheduler(opt, "warmup_cosine", warmup_steps=10, total_steps=1000)
        opt.step(); sch.step()
        assert opt.param_groups[0]["lr"] < 1e-3                 # warming up
        opt2 = torch.optim.AdamW(lin.parameters(), lr=1e-3)
        sch2 = build_scheduler(opt2, "none", warmup_steps=10, total_steps=1000)
        opt2.step(); sch2.step()
        assert opt2.param_groups[0]["lr"] == pytest.approx(1e-3)
        with pytest.raises(ValueError):
            build_scheduler(opt, "bogus", 1, 10)
        cfg = SkyFlowConfig()
        assert cfg.training.scheduler == "warmup_cosine" and cfg.training.warmup_steps == 200
        assert cfg.training.tbptt_detach is False
