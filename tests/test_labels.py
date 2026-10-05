"""Ground-truth label definition (lookahead vs. legacy instantaneous)."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from skyflow.data.urbanair500 import UrbanAir500, scenario_seed
from tests._helpers import straight_line_log as _straight_line_log


def _sim(mode, N):
    return UrbanAir500(num_uavs=N, seed=0, label_mode=mode, lookahead_s=30.0,
                       num_sectors=4, num_weather_cells=4, num_restricted_zones=1)


class TestLookaheadLabels:
    def test_head_on_pair_is_positive_with_correct_ttc(self):
        # 400 m apart, closing at 40 m/s -> within 10 m after ~9.75 s
        log = _straight_line_log(
            p0=[[0, 0, 50], [400, 0, 50], [0, 2000, 50]],
            v=[[20, 0, 0], [-20, 0, 0], [0, 0, 0]],
            n_total=400,
        )
        ev = _sim("lookahead", 3).label(log, epoch=0)
        pairs = {(e.uav_i, e.uav_j): e for e in ev}
        assert (0, 1) in pairs
        assert (0, 2) not in pairs and (1, 2) not in pairs
        assert pairs[(0, 1)].time_to_conflict == pytest.approx(9.8, abs=0.15)
        assert pairs[(0, 1)].min_separation_h < 10.0

    def test_conflict_beyond_window_is_negative(self):
        # 2000 m apart closing at 40 m/s -> LoS at ~50 s > 30 s window
        log = _straight_line_log(
            p0=[[0, 0, 50], [2000, 0, 50]], v=[[20, 0, 0], [-20, 0, 0]], n_total=700,
        )
        assert _sim("lookahead", 2).label(log, 0) == []

    def test_vertical_separation_prevents_conflict(self):
        log = _straight_line_log(
            p0=[[0, 0, 50], [400, 0, 54]], v=[[20, 0, 0], [-20, 0, 0]], n_total=400,
        )
        assert _sim("lookahead", 2).label(log, 0) == []

    def test_window_truncated_at_log_end(self):
        log = _straight_line_log(
            p0=[[0, 0, 50], [400, 0, 50]], v=[[20, 0, 0], [-20, 0, 0]], n_total=50,
        )
        # only 5 s of future available -> no LoS yet
        assert _sim("lookahead", 2).label(log, 0) == []

    def test_instantaneous_mode_is_legacy(self):
        log = _straight_line_log(
            p0=[[0, 0, 50], [400, 0, 50]], v=[[20, 0, 0], [-20, 0, 0]], n_total=400,
        )
        sim = _sim("instantaneous", 2)
        assert sim.label(log, 0) == []            # 400 m apart now
        assert len(sim.label(log, 100)) == 1      # 0 m apart at t = 10 s

    def test_simulate_scenario_runs_physics_beyond_horizon(self):
        sim = UrbanAir500(num_uavs=10, grid_size=600.0, seed=3, label_mode="lookahead",
                          lookahead_s=30.0, num_sectors=4, num_weather_cells=4,
                          num_restricted_zones=1)
        plans = sim.generate_flight_plans(10)
        log = sim.run_physics(plans, 2.0, extra_seconds=30.0)
        assert log.n_epochs == 20 and log.n_total == 320
        states = list(sim.simulate_scenario(2.0, plans, label_every=10))
        assert len(states) == 20
        assert all(c == [] for i, (s, c) in enumerate(states) if i % 10 != 0)


class TestDeterminism:
    def test_scenario_seed_is_stable(self):
        assert scenario_seed(42, "train", 3) == scenario_seed(42, "train", 3)
        assert scenario_seed(42, "train", 3) != scenario_seed(42, "val", 3)
        assert scenario_seed(42, "train", 3) != scenario_seed(42, "train", 4)

    def test_generate_dataset_reproducible(self):
        kw = dict(num_uavs=12, grid_size=600.0, seed=5, num_sectors=4,
                  num_weather_cells=4, num_restricted_zones=1)
        a = UrbanAir500(**kw).generate_dataset("val", 1, 2.0)
        b = UrbanAir500(**kw).generate_dataset("val", 1, 2.0)
        assert len(a) == len(b) == 2
        for (sa, la), (sb, lb) in zip(a, b):
            assert np.array_equal(sa.node_features.numpy(), sb.node_features.numpy())
            assert np.array_equal(la.numpy(), lb.numpy())
            assert np.array_equal(sa.conflict_pairs.numpy(), sb.conflict_pairs.numpy())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
