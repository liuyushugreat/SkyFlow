"""S8d: filed flight-plan context and the plan-aware CPA baseline.

  * ``filed_plan_context`` projects the *observed* position onto the filed
    polyline and advances along it - checked against hand-computed values on
    a two-segment route, including the clamp at the route end;
  * it depends only on the filed plan and the observed position (tampering
    with the true future trajectory leaves it unchanged), and is zero for
    non-cooperative UAVs;
  * the simulator stores filed (not flown) waypoints, so nonconforming UAVs
    get plan context that differs from their flown path;
  * TKGBuilder writes the 12 columns after the 20 observed features, and
    ``features.plan_context: false`` reproduces the 20-d set and a different
    cache key;
  * ``PlanCPARule`` flags a pair whose filed routes cross even when linear
    extrapolation of the current velocities does not, and falls back to
    linear extrapolation for UAVs without plan context.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from skyflow.baselines.cpa_rule import CPARule, PlanCPARule
from skyflow.baselines.registry import REGISTRY
from skyflow.config import SkyFlowConfig
from skyflow.data.cache import data_signature
from skyflow.data.tkg_builder import PLAN_CONTEXT_FEATURES, TKGBuilder, TKGSnapshot, uav_feature_names
from skyflow.models.conflict_head import build_pair_edge_features, plan_column
from skyflow.data.urbanair500 import (
    CAUSE_CODES, TruthLog, UAVFlightPlan, UrbanAir500, filed_plan_context, pack_filed_plans,
)


def _log_with_plans(plans, n, cooperative=None):
    wps, cnt, spd, st = pack_filed_plans(plans, n)
    log = TruthLog.__new__(TruthLog)
    log.plan_waypoints, log.plan_num_waypoints, log.plan_cruise_speed, log.plan_start_time = wps, cnt, spd, st
    log.cooperative = cooperative
    return log


class TestFiledPlanContext:
    def test_two_segment_route_hand_computed(self):
        # route: (0,0,50) -> (100,0,50) -> (100,100,50), cruise 10 m/s
        plan = UAVFlightPlan(0, np.array([[0, 0, 50], [100, 0, 50], [100, 100, 50]], np.float32), 1, 10.0, 0.0)
        log = _log_with_plans([plan], 1)
        p = np.array([[40.0, 5.0, 50.0]])          # 5 m off-track, abeam s = 40 m of segment 1
        ctx = filed_plan_context(log, p, horizons_s=(10.0, 20.0, 30.0))
        assert ctx.shape == (1, 12)
        np.testing.assert_allclose(ctx[0, 0:3], [60.0, -5.0, 0.0], atol=1e-4)     # next waypoint (100,0,50)
        # planned path: p -> (100,0) [60.21 m] -> (100,100); +10 s = 100 m -> 39.79 m up the 2nd leg
        leg1 = np.hypot(60.0, 5.0)
        np.testing.assert_allclose(ctx[0, 3:6], [60.0, (100.0 - leg1) - 5.0, 0.0], atol=1e-3)
        # path length is 160.21 m: +20 s (200 m) and +30 s are clamped to the route end (100,100)
        np.testing.assert_allclose(ctx[0, 6:9], [60.0, 95.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(ctx[0, 9:12], [60.0, 95.0, 0.0], atol=1e-4)

    def test_heading_disambiguates_revisited_hub(self):
        # out-and-back route through the same hub: (0,0)->(200,0)->(0,0). A UAV at (100,0)
        # flying +x is on leg 1 (next waypoint (200,0)); flying -x it is on leg 2 (next (0,0)).
        plan = UAVFlightPlan(0, np.array([[0, 0, 50], [200, 0, 50], [0, 0, 50]], np.float32), 1, 10.0, 0.0)
        log = _log_with_plans([plan], 1)
        p = np.array([[100.0, 0.0, 50.0]])
        fwd = filed_plan_context(log, p, 0.0, np.array([[10.0, 0.0, 0.0]]))
        back = filed_plan_context(log, p, 0.0, np.array([[-10.0, 0.0, 0.0]]))
        np.testing.assert_allclose(fwd[0, 0:3], [100.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(back[0, 0:3], [-100.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(fwd[0, 9:12], [-100.0, 0.0, 0.0], atol=1e-4)   # +30 s: 300 m -> back at (0,0)

    def test_independent_of_true_future_and_zero_for_noncooperative(self):
        plan0 = UAVFlightPlan(0, np.array([[0, 0, 50], [300, 0, 50]], np.float32), 1, 10.0, 0.0)
        plan1 = UAVFlightPlan(1, np.array([[0, 100, 60], [300, 100, 60]], np.float32), 1, 10.0, 0.0,
                              cause="noncooperative", cooperative=False)
        log = _log_with_plans([plan0, plan1], 2, cooperative=np.array([True, False]))
        p = np.array([[50.0, 0.0, 50.0], [50.0, 100.0, 60.0]])
        a = filed_plan_context(log, p)
        # 'future' truth is not an input at all; re-evaluating with another
        # observed position only changes the result through that position
        b = filed_plan_context(log, p + np.array([[1.0, 0, 0], [0, 0, 0]]))
        assert not np.allclose(a[0], b[0]) and np.allclose(a[1], b[1])
        assert np.all(a[1] == 0.0)                          # non-cooperative -> no plan known
        np.testing.assert_allclose(a[0, 3:6], [100.0, 0.0, 0.0], atol=1e-4)

    def test_simulator_stores_filed_not_flown_plan(self):
        sim = UrbanAir500(num_uavs=30, grid_size=800.0, seed=3)
        plans = sim.generate_flight_plans(30)
        log = sim.run_physics(plans, 2.0)
        assert log.plan_waypoints is not None and log.plan_waypoints.shape[0] == 30
        for p in plans:
            k = len(p.waypoints)
            np.testing.assert_allclose(log.plan_waypoints[p.uav_id, :k], p.waypoints, atol=1e-5)
        state = sim.observe(log, 10)
        assert state.uav_plan_context is not None and state.uav_plan_context.shape == (30, 12)
        if log.cooperative is not None and (~log.cooperative).any():
            assert np.all(state.uav_plan_context[~log.cooperative] == 0)


class TestBuilderAndConfig:
    def test_feature_layout(self):
        names = uav_feature_names(True, True)
        assert names[20:] == PLAN_CONTEXT_FEATURES and len(names) == 32

    def test_builder_writes_plan_columns(self):
        sim = UrbanAir500(num_uavs=20, grid_size=600.0, seed=5)
        log = sim.run_physics(sim.generate_flight_plans(20), 1.0)
        state = sim.observe(log, 5)
        snap = TKGBuilder().build(state)
        n = snap.num_uavs
        np.testing.assert_allclose(snap.node_features[:n, 20:32].numpy(), state.uav_plan_context, atol=1e-4)
        assert torch.all(snap.node_features[n:, 20:32] == 0)     # context nodes untouched
        snap0 = TKGBuilder(plan_context=False).build(state)
        assert torch.equal(snap0.node_features[:n], snap.node_features[:n, :20])

    def test_switch_changes_cache_key_and_dims(self):
        cfg = SkyFlowConfig()
        sig_on = data_signature(cfg, "train")
        assert cfg.uav_feature_dim() == 32
        cfg.features.plan_context = False
        assert data_signature(cfg, "train") != sig_on
        assert cfg.uav_feature_dim() == 20
        assert cfg.make_builder().feature_dim == 20


def _snapshot_from(pos, vel, plan_ctx):
    """Two-UAV snapshot with the given observed state and plan context (12-d)."""
    n = pos.shape[0]
    nf = torch.zeros(n, 32)
    nf[:, 0:3] = torch.tensor(pos, dtype=torch.float32)
    nf[:, 3:6] = torch.tensor(vel, dtype=torch.float32)
    nf[:, 20:32] = torch.tensor(plan_ctx, dtype=torch.float32)
    return TKGSnapshot(
        node_features=nf, node_types=torch.zeros(n, dtype=torch.long),
        edge_indices={}, edge_deltas={}, num_uavs=n, num_nodes=n,
        conflict_pairs=torch.tensor([[0], [1]]),
        feature_names=uav_feature_names(True, True),
    )


class TestPlannedPairGeometry:
    def _two(self):
        # UAV0 at origin, plan: +x 100 m per 10 s; UAV1 at (300, 0, 50) hovering, plan: -x 100 m per 10 s
        pos = np.array([[0.0, 0.0, 50.0], [300.0, 0.0, 52.0]])
        vel = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        ctx = np.zeros((2, 12))
        for k in range(3):
            ctx[0, 3 + 3 * k: 6 + 3 * k] = [100.0 * (k + 1), 0.0, 0.0]
            ctx[1, 3 + 3 * k: 6 + 3 * k] = [-100.0 * (k + 1), 0.0, 0.0]
        return pos, vel, ctx

    def test_planned_separations(self):
        pos, vel, ctx = self._two()
        snap = _snapshot_from(pos, vel, ctx)
        e = build_pair_edge_features(snap.node_features, snap.conflict_pairs, None, mode="geometry_plan",
                                     plan_col0=plan_column(snap.feature_names))
        assert e.shape == (1, 20)
        g = e[0, 12:]
        # planned horizontal separation: 300 -> 100 (+10 s) -> 100 (+20 s, crossed) -> 300 (+30 s)
        assert g[0].item() == pytest.approx(1.0)       # +10 s: |300-200| = 100 m
        assert g[2].item() == pytest.approx(1.0)       # +20 s: |-100| = 100 m
        assert g[4].item() == pytest.approx(3.0)       # +30 s: 300 m
        assert g[1].item() == pytest.approx(0.2)       # dz 2 m / 10
        assert g[6].item() == pytest.approx(1.0)       # min planned horizontal separation
        assert g[7].item() == 0.0                      # both have plans

    def test_no_plan_fallback_and_flag(self):
        pos, vel, ctx = self._two()
        ctx[1] = 0.0                                   # UAV1: no plan context (non-cooperative)
        snap = _snapshot_from(pos, vel, ctx)
        e = build_pair_edge_features(snap.node_features, snap.conflict_pairs, None, mode="geometry_plan",
                                     plan_col0=plan_column(snap.feature_names))
        g = e[0, 12:]
        assert g[7].item() == 1.0
        # fallback uses observed velocity (-10 m/s) -> same planned separations as before
        assert g[0].item() == pytest.approx(1.0) and g[4].item() == pytest.approx(3.0)

    def test_geometry_plan_requires_plan_columns(self):
        pos, vel, ctx = self._two()
        snap = _snapshot_from(pos, vel, ctx)
        with pytest.raises(ValueError):
            build_pair_edge_features(snap.node_features, snap.conflict_pairs, None, mode="geometry_plan")
        # and the ablation config switches both the UAV features and the pair mode off
        from skyflow.experiments.methods import method_config
        cfg = method_config("abl_no_plan", SkyFlowConfig())
        assert cfg.features.plan_context is False and cfg.features.pair_edge_features == "geometry"
        assert cfg.uav_feature_dim() == 20

    def test_learned_scorers_run_with_geometry_plan(self):
        sim = UrbanAir500(num_uavs=20, grid_size=600.0, seed=5)
        log = sim.run_physics(sim.generate_flight_plans(20), 1.0, extra_seconds=30.0)
        snap = TKGBuilder().build(sim.observe(log, 5))
        snap.conflict_pairs = snap.candidate_pairs
        from skyflow.baselines.gat_static import GATStatic
        from skyflow.baselines.lstm_pair import LSTMPair
        for cls in (GATStatic, LSTMPair):
            m = cls(input_dim=32, pair_edge_features="geometry_plan"); m.eval()
            out = m(snap)
            assert out.shape == (snap.conflict_pairs.size(1),) and torch.isfinite(out).all()


class TestPlanCPARule:
    def test_detects_planned_turn_that_extrapolation_misses(self):
        # UAV0 flies +x at 10 m/s but its filed plan turns towards (100, 200) after 10 s;
        # UAV1 hovers at (100, 200, 50). Linear extrapolation: no conflict. Plan: conflict.
        pos = np.array([[0.0, 0.0, 50.0], [100.0, 200.0, 50.0]])
        vel = np.array([[10.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        ctx0 = np.zeros(12)
        ctx0[0:3] = [100.0, 0.0, 0.0]           # next waypoint
        ctx0[3:6] = [100.0, 0.0, 0.0]           # +10 s at (100, 0)
        ctx0[6:9] = [100.0, 100.0, 0.0]         # +20 s at (100, 100)
        ctx0[9:12] = [100.0, 200.0, 0.0]        # +30 s at (100, 200) -> meets UAV1
        ctx1 = np.zeros(12); ctx1[3:6] = ctx1[6:9] = ctx1[9:12] = [0.0, 0.0, 0.0]
        # UAV1 has 'no plan' (all zero) -> fallback to linear extrapolation with v = 0 (hover)
        snap = _snapshot_from(pos, vel, np.stack([ctx0, ctx1]))
        plain = CPARule(window_s=30.0, h_thresh=10.0, v_thresh=3.0)
        plan = PlanCPARule(window_s=30.0, h_thresh=10.0, v_thresh=3.0)
        assert plain.predict(snap).tolist() == [0.0]
        assert plan.predict(snap).tolist() == [1.0]

    def test_reduces_to_cpa_when_plan_is_straight(self):
        pos = np.array([[0.0, 0.0, 50.0], [300.0, 20.0, 55.0]])
        vel = np.array([[10.0, 0.0, 0.0], [-10.0, 0.0, 0.0]])
        ctx = np.zeros((2, 12))
        for k, tau in enumerate((10.0, 20.0, 30.0)):
            ctx[:, 3 + 3 * k: 6 + 3 * k] = vel * tau
        ctx[:, 0:3] = vel * 30.0
        snap = _snapshot_from(pos, vel, ctx)
        for h, v in ((10.0, 3.0), (30.0, 10.0), (15.0, 8.0)):
            a = CPARule(30.0, h, v).predict(snap).tolist()
            b = PlanCPARule(30.0, h, v).predict(snap).tolist()
            assert a == b

    def test_requires_plan_features(self):
        snap = _snapshot_from(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 12)))
        snap.feature_names = uav_feature_names(True, False)
        with pytest.raises(ValueError):
            PlanCPARule().predict(snap)

    def test_registry_and_val_search(self):
        cfg = SkyFlowConfig()
        rule = REGISTRY["Plan-CPA"].factory(cfg, torch.device("cpu"))
        assert isinstance(rule, PlanCPARule) and rule.threshold_mode == cfg.baselines.cpa_rule_thresholds
        sim = UrbanAir500(num_uavs=40, grid_size=400.0, seed=9)
        log = sim.run_physics(sim.generate_flight_plans(40), 3.0, extra_seconds=30.0)
        b = TKGBuilder()
        data = []
        for e in (0, 10, 20):
            snap = b.build(sim.observe(log, e))
            snap.conflict_pairs = snap.candidate_pairs
            data.append((snap, torch.zeros(snap.conflict_pairs.size(1))))
        rule.threshold_mode = "val_search"
        h, v = rule.fit(data)
        assert (h, v) in {(hh, vv) for hh in rule.h_grid for vv in rule.v_grid}
