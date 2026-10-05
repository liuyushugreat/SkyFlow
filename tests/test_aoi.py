"""S3: δ as Age of Information (AoI) with the ADS-B observation model."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
import torch

from skyflow.data.tkg_builder import TKGBuilder
from skyflow.data.urbanair500 import UrbanAir500, ObservationParams
from tests._helpers import straight_line_log

DT = 0.1


def _sim(**kw):
    base = dict(num_uavs=4, seed=0, num_sectors=4, num_weather_cells=4,
                num_restricted_zones=1, grid_size=1000.0, observation_model="adsb",
                adsb_latency_range=(0.0, 0.0), packet_loss=0.0)
    base.update(kw)
    return UrbanAir500(**base)


def _close_pair_log(n_total=40):
    # UAVs 0,1 converge (approach edge); 2,3 far away but also converging
    return straight_line_log(
        p0=[[500, 500, 80], [560, 500, 80], [100, 100, 80], [160, 100, 80]],
        v=[[5, 0, 0], [-5, 0, 0], [5, 0, 0], [-5, 0, 0]],
        n_total=n_total, dt=DT,
    )


def _uav_edge_deltas(snap):
    """δ of every UAV–UAV edge (relation 'approaches' == index 0), keyed (src, dst)."""
    out = {}
    if 0 in snap.edge_indices:
        ei = snap.edge_indices[0].numpy()
        ed = snap.edge_deltas[0].numpy()
        for k in range(ei.shape[1]):
            out[(int(ei[0, k]), int(ei[1, k]))] = float(ed[k])
    return out


class TestAoIZeroWhenFresh:
    def test_no_latency_no_loss_gives_zero_delta(self):
        sim = _sim()
        log = _close_pair_log()
        builder = TKGBuilder(delta_mode="aoi")
        for e in range(0, 30, 3):
            snap = builder.build(sim.observe(log, e))
            assert torch.all(snap.uav_aoi == 0)
            for d in _uav_edge_deltas(snap).values():
                assert d == 0.0
        assert len(_uav_edge_deltas(snap)) > 0, "approach edges expected"

    def test_legacy_observation_has_zero_aoi(self):
        sim = _sim(observation_model="legacy")
        log = _close_pair_log()
        snap = TKGBuilder(delta_mode="aoi").build(sim.observe(log, 10))
        assert torch.all(snap.uav_aoi == 0)


class TestAoIUnderLoss:
    def test_three_consecutive_lost_frames_increase_delta_per_frame(self):
        sim = _sim(packet_loss=0.5)
        log = _close_pair_log()
        # UAV 0 loses the reports generated at epochs 5,6,7 (u < p)
        log.loss_u[5:8, 0] = 0.0
        builder = TKGBuilder(delta_mode="aoi")

        expected_aoi0 = {4: 0.0, 5: 0.1, 6: 0.2, 7: 0.3, 8: 0.0}
        for e, want in expected_aoi0.items():
            snap = builder.build(sim.observe(log, e))
            aoi = snap.uav_aoi.numpy()
            assert aoi[0] == pytest.approx(want, abs=1e-6), (e, aoi)
            assert aoi[1] == pytest.approx(0.0, abs=1e-6)
            deltas = _uav_edge_deltas(snap)
            assert deltas[(0, 1)] == pytest.approx(want, abs=1e-6)   # δ_ij = max(AoI_i, AoI_j)
            assert deltas[(1, 0)] == pytest.approx(want, abs=1e-6)
            assert deltas[(2, 3)] == pytest.approx(0.0, abs=1e-6)    # untouched pair

    def test_lost_report_freezes_observed_state(self):
        sim = _sim(packet_loss=0.5)
        log = _close_pair_log()
        log.loss_u[5:8, 0] = 0.0
        s4 = sim.observe(log, 4)
        s6 = sim.observe(log, 6)
        assert np.allclose(s6.uav_positions[0], s4.uav_positions[0])      # frozen
        assert not np.allclose(s6.uav_positions[1], s4.uav_positions[1])  # others move
        assert s6.uav_last_rx_time[0] == pytest.approx(4 * DT)


class TestAoIUnderLatency:
    def test_constant_latency_gives_constant_aoi(self):
        sim = _sim(adsb_latency_range=(0.5, 0.5))
        log = _close_pair_log()
        builder = TKGBuilder(delta_mode="aoi")
        snap = builder.build(sim.observe(log, 20))
        assert torch.allclose(snap.uav_aoi, torch.full((4,), 0.5))
        # observed position of UAV 1 is the truth 5 frames ago
        s = sim.observe(log, 20)
        assert np.allclose(s.uav_positions[1], log.positions[15, 1])

    def test_before_first_report_departure_state_is_used(self):
        sim = _sim(adsb_latency_range=(1.0, 1.0))
        log = _close_pair_log()
        s = sim.observe(log, 3)                      # 0.3 s < 1.0 s latency
        assert np.allclose(s.uav_positions, log.positions[0])
        assert np.allclose(s.uav_last_rx_time, 0.0)

    def test_observation_params_override_without_resim(self):
        sim = _sim(adsb_latency_range=(0.0, 0.0))
        log = _close_pair_log()
        fresh = sim.observe(log, 20)
        stale = sim.observe(log, 20, ObservationParams(adsb_latency_s=1.0, packet_loss=0.0))
        assert np.allclose(fresh.uav_positions[1], log.positions[20, 1])
        assert np.allclose(stale.uav_positions[1], log.positions[10, 1])


class TestEnvironmentAoI:
    def test_env_edge_delta_is_source_age(self):
        sim = _sim(weather_update_s=5.0, registry_update_s=10.0)
        plans = sim.generate_flight_plans(4)
        log = sim.run_physics(plans, 3.0, extra_seconds=0.0)
        builder = TKGBuilder(delta_mode="aoi")
        snap = builder.build(sim.observe(log, 23))                    # t = 2.3 s
        wx_idx = builder.relations["is_downwind_of"]
        rz_idx = builder.relations["is_restricted_by"]
        if wx_idx in snap.edge_deltas:
            assert torch.allclose(snap.edge_deltas[wx_idx],
                                  torch.full_like(snap.edge_deltas[wx_idx], 2.3), atol=1e-5)
        if rz_idx in snap.edge_deltas:
            assert torch.allclose(snap.edge_deltas[rz_idx],
                                  torch.full_like(snap.edge_deltas[rz_idx], 2.3), atol=1e-5)
        snap = builder.build(sim.observe(log, 7))                     # t = 0.7 s
        if wx_idx in snap.edge_deltas:
            assert torch.allclose(snap.edge_deltas[wx_idx],
                                  torch.full_like(snap.edge_deltas[wx_idx], 0.7), atol=1e-5)


class TestLegacyDeltaStillAvailable:
    def test_legacy_delta_counts_time_since_edge_seen(self):
        sim = _sim()
        log = _close_pair_log()
        builder = TKGBuilder(delta_mode="legacy")
        d0 = _uav_edge_deltas(builder.build(sim.observe(log, 0)))
        d1 = _uav_edge_deltas(builder.build(sim.observe(log, 10)))
        assert d0[(0, 1)] == 0.0
        assert d1[(0, 1)] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
