"""S6: simulator parametrisation (observation conditions, telemetry_only,
re-evaluation without re-simulation, density preset)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGBuilder, ENTITY_TYPES
from skyflow.data.urbanair500 import UrbanAir500, ObservationParams, CEP_TO_SIGMA, VERTICAL_SIGMA_FACTOR


def _sim(**kw):
    base = dict(num_uavs=40, grid_size=1000.0, seed=11, num_sectors=4,
                num_weather_cells=4, num_restricted_zones=2, cause_mix=None)
    base.update(kw)
    return UrbanAir500(**base)


class TestObservationConditions:
    def test_full_packet_loss_freezes_all_states(self):
        sim = _sim(packet_loss=1.0)
        log = sim.simulate_logs("val", 1, 5.0)[0]
        s0 = sim.observe(log, 0)
        for e in (10, 25, 49):
            s = sim.observe(log, e)
            assert np.array_equal(s.uav_positions, s0.uav_positions)
            assert np.array_equal(s.uav_velocities, s0.uav_velocities)
            assert np.all(s.uav_last_rx_time == 0.0)
        # truth did move
        assert not np.allclose(log.positions[49], log.positions[0])

    def test_zero_latency_zero_loss_differs_from_truth_only_by_gps_noise(self):
        sim = _sim(adsb_latency_range=(0.0, 0.0), packet_loss=0.0, gps_cep=2.5)
        log = sim.simulate_logs("val", 1, 5.0)[0]
        sigma_h = 2.5 * CEP_TO_SIGMA
        for e in (0, 17, 49):
            s = sim.observe(log, e)
            diff = s.uav_positions - log.positions[e]
            expected = log.gps_noise_unit[e] * np.array([sigma_h, sigma_h, sigma_h * VERTICAL_SIGMA_FACTOR], np.float32)
            assert np.allclose(diff, expected, atol=5e-3)     # float32 at ~1e3 m scale
            assert np.array_equal(s.uav_velocities, log.velocities[e])
            assert np.all(s.uav_last_rx_time == pytest.approx(e * sim.dt))
        # with CEP -> 0 the observation equals the truth exactly
        s = sim.observe(log, 17, ObservationParams(adsb_latency_s=0.0, packet_loss=0.0, gps_cep_m=0.0))
        assert np.allclose(s.uav_positions, log.positions[17], atol=1e-5)

    def test_config_plumbing(self):
        cfg = SkyFlowConfig()
        cfg.sim.adsb_latency_s = [0.2, 0.4]
        cfg.sim.packet_loss = 0.1
        cfg.sim.gps_cep_m = 4.0
        cfg.sim.density_preset = "legacy"
        sim = cfg.make_simulator(num_uavs=10)
        assert sim.obs_params.latency_range() == (0.2, 0.4)
        assert sim.obs_params.packet_loss == 0.1
        assert sim.obs_params.gps_cep_m == 4.0
        assert sim.density_preset == "legacy"
        cfg.sim.adsb_latency_s = 0.7
        assert cfg.make_simulator(num_uavs=10).obs_params.latency_range() == (0.7, 0.7)


class TestTelemetryOnly:
    def test_graph_has_only_uav_nodes_and_approach_edges(self):
        sim = _sim()
        log = sim.simulate_logs("val", 1, 5.0)[0]
        state = sim.observe(log, 30)
        full = TKGBuilder(input_set="full").build(state)
        tel = TKGBuilder(input_set="telemetry_only").build(state)
        assert tel.num_nodes == tel.num_uavs == 40
        assert set(tel.node_types.tolist()) == {ENTITY_TYPES["uav"]}
        assert set(tel.edge_indices) <= {0}                       # only 'approaches'
        assert full.num_nodes > full.num_uavs
        assert torch.equal(tel.node_features, full.node_features[:40])
        if 0 in full.edge_indices:
            assert torch.equal(tel.edge_indices[0], full.edge_indices[0])
        assert tel.relation_names == full.relation_names      # vocab unchanged (same model shapes)

    def test_config_plumbing(self):
        cfg = SkyFlowConfig()
        cfg.features.input_set = "telemetry_only"
        assert cfg.make_builder().telemetry_only
        with pytest.raises(ValueError):
            TKGBuilder(input_set="uav_only")


class TestReevaluateWithoutResim:
    def test_same_logs_new_obs_params_same_labels_different_features(self):
        sim = _sim(adsb_latency_range=(0.5, 1.2))
        logs = sim.simulate_logs("test", 2, 6.0)
        base = sim.dataset_from_logs(logs)
        harsh = sim.dataset_from_logs(
            logs, obs_params=ObservationParams(adsb_latency_s=(2.0, 3.0), packet_loss=0.3, gps_cep_m=10.0)
        )
        assert len(base) == len(harsh) > 0
        n_diff = 0
        for (sb, lb), (sh, lh) in zip(base, harsh):
            # candidate set may change (observed positions differ); positives are conserved
            assert int(lb.sum()) + sb.num_missed_positives == int(lh.sum()) + sh.num_missed_positives
            n_diff += int(not torch.equal(sb.node_features, sh.node_features))
        assert n_diff > 0
        # dataset_from_logs == generate_dataset for the default conditions
        direct = sim.generate_dataset("test", 2, 6.0)
        for (a, la), (b, lb) in zip(base, direct):
            assert torch.equal(a.node_features, b.node_features)
            assert torch.equal(la, lb)

    def test_infrastructure_is_restored_per_scenario(self):
        sim = _sim(num_restricted_zones=3)
        logs = sim.simulate_logs("val", 2, 2.0)
        rz0 = logs[0].infrastructure["restricted_zones"]
        rz1 = logs[1].infrastructure["restricted_zones"]
        assert not np.array_equal(rz0, rz1)
        sim.dataset_from_logs([logs[0]])
        assert np.array_equal(sim.restricted_zones, rz0)
        sim.dataset_from_logs([logs[1]])
        assert np.array_equal(sim.restricted_zones, rz1)


class TestDensityPreset:
    def test_dense_uses_altitude_layers_and_hubs(self):
        sim = _sim(density_preset="dense", num_uavs=60)
        plans = sim.generate_flight_plans(60)
        z = np.concatenate([p.waypoints[:, 2] for p in plans])
        layers = np.array([60.0, 80.0, 100.0, 120.0])
        assert np.all(np.min(np.abs(z[:, None] - layers[None, :]), axis=1) <= 1.0 + 1e-4)
        # each plan stays in one layer
        for p in plans:
            assert np.ptp(p.waypoints[:, 2]) <= 2.0 + 1e-4
        # many waypoints sit on corridor hubs
        hubs = sim.corridor_nodes[:, :2]
        wps = np.concatenate([p.waypoints[1:, :2] for p in plans])
        d = np.min(np.linalg.norm(wps[:, None, :] - hubs[None, :, :], axis=-1), axis=1)
        assert np.mean(d < 45.0) > 0.4

    def test_legacy_preset_is_unchanged_behaviour(self):
        sim = _sim(density_preset="legacy", num_uavs=60)
        plans = sim.generate_flight_plans(60)
        z = np.concatenate([p.waypoints[:, 2] for p in plans])
        assert np.ptp(z) > 60.0            # continuous altitudes over the band
        with pytest.raises(ValueError):
            _sim(density_preset="very_dense")

    def test_dense_gives_more_positives_than_legacy(self):
        pos = {}
        for preset in ("dense", "legacy"):
            sim = UrbanAir500(num_uavs=150, grid_size=5000.0, seed=5, density_preset=preset,
                              num_sectors=4, num_weather_cells=4, num_restricted_zones=2)
            data = sim.generate_dataset("val", 1, 20.0, candidates="all")
            pos[preset] = sum(int(l.sum()) for _, l in data)
        assert pos["dense"] > 3 * max(pos["legacy"], 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
