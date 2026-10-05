"""S6 item 2: synthetic conflict-cause injection and attribution."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from skyflow.config import SkyFlowConfig
from skyflow.data.tkg_builder import TKGBuilder
from skyflow.data.urbanair500 import (
    UrbanAir500, CAUSES, CAUSE_CODES, NONCOOP_EXTRA_LATENCY_S, TruthLog,
)
from tests._helpers import straight_line_log


def _sim(**kw):
    base = dict(num_uavs=80, grid_size=1500.0, seed=21, num_sectors=4,
                num_weather_cells=4, num_restricted_zones=2)
    base.update(kw)
    return UrbanAir500(**base)


def test_cause_mix_none_leaves_plans_unchanged():
    a = _sim(cause_mix=None).generate_flight_plans(80)
    b = _sim(cause_mix="default").generate_flight_plans(80)
    assert all(p.cause == "planned_crossing" for p in a)
    assert len({p.cause for p in b}) > 1
    # ordinary UAVs in the injected run have exactly the plans of the clean run
    for pa, pb in zip(a, b):
        if pb.cause in ("planned_crossing", "wind_deviation", "nonconforming", "noncooperative"):
            assert np.array_equal(pa.waypoints, pb.waypoints)
            assert pa.cruise_speed == pb.cruise_speed and pa.start_time == pb.start_time


def test_each_cause_has_its_mechanism():
    plans = _sim(num_uavs=400, cause_mix={c: 0.2 for c in CAUSES}).generate_flight_plans(400)
    by = {c: [p for p in plans if p.cause == c] for c in CAUSES}
    assert all(len(v) > 10 for v in by.values())
    for p in by["wind_deviation"]:
        assert p.wind_drift is not None and 2.0 <= np.linalg.norm(p.wind_drift) <= 4.0
    for p in by["nonconforming"]:
        assert p.deviations is not None and p.deviations.shape == p.waypoints.shape
    assert any(np.abs(p.deviations).max() >= 100.0 for p in by["nonconforming"])
    for p in by["priority_insertion"]:
        assert p.priority == 0 and 10.0 <= p.start_time <= 30.0 and len(p.waypoints) == 3
        assert 20.0 <= p.cruise_speed <= 25.0
    for p in by["noncooperative"]:
        assert p.cooperative is False
    for p in by["planned_crossing"]:
        assert p.cooperative and p.wind_drift is None and p.deviations is None


def test_invalid_cause_mix_rejected():
    with pytest.raises(ValueError):
        _sim(cause_mix={"meteor_strike": 1.0})
    with pytest.raises(ValueError):
        _sim(cause_mix={"planned_crossing": 0.0})
    sim = _sim(cause_mix={"planned_crossing": 3, "noncooperative": 1})
    assert sim.cause_mix["planned_crossing"] == pytest.approx(0.75)
    assert sim.cause_mix["noncooperative"] == pytest.approx(0.25)


def test_noncooperative_uav_has_larger_aoi():
    sim = _sim(cause_mix={"planned_crossing": 0.5, "noncooperative": 0.5}, adsb_latency_range=(0.0, 0.0))
    log = sim.simulate_logs("val", 1, 6.0)[0]
    assert log.cooperative is not None and (~log.cooperative).sum() > 0
    snap = TKGBuilder().build(sim.observe(log, 55))
    aoi = snap.uav_aoi.numpy()
    assert np.all(aoi[~log.cooperative] == pytest.approx(NONCOOP_EXTRA_LATENCY_S, abs=1e-5))
    assert np.all(aoi[log.cooperative] == 0.0)


def test_wind_drift_moves_uav_off_filed_track():
    sim = _sim(cause_mix=None)
    plans = sim.generate_flight_plans(80)
    # make UAV 0 a long straight flight, then compare with/without drift
    plans[0].waypoints = np.array([[200, 750, 100], [1300, 750, 100]], np.float32)
    plans[0].start_time = 0.0
    sim.rng = np.random.RandomState(0)
    clean = sim.run_physics(plans, 20.0)
    plans[0].cause = "wind_deviation"
    plans[0].wind_drift = np.array([0.0, 3.0, 0.0], np.float32)
    sim.rng = np.random.RandomState(0)          # identical wind / battery draws
    drifted = sim.run_physics(plans, 20.0)
    assert drifted.positions[-1, 0, 1] - clean.positions[-1, 0, 1] > 10.0
    assert np.allclose(drifted.positions[-1, 1:], clean.positions[-1, 1:])  # others unaffected


def test_conflict_cause_attribution_priority():
    log = straight_line_log(
        p0=[[0, 0, 80], [100, 0, 80], [0, 500, 80], [100, 500, 80]],
        v=[[5, 0, 0], [-5, 0, 0], [5, 0, 0], [-5, 0, 0]], n_total=400,
    )
    log.causes = np.array([CAUSE_CODES["wind_deviation"], CAUSE_CODES["noncooperative"],
                           CAUSE_CODES["planned_crossing"], CAUSE_CODES["planned_crossing"]], np.int8)
    sim = _sim(num_uavs=4, cause_mix=None, lookahead_s=30.0)
    events = {(e.uav_i, e.uav_j): e for e in sim.label(log, 0)}
    assert events[(0, 1)].cause == "noncooperative"     # higher-ranked cause wins
    assert events[(2, 3)].cause == "planned_crossing"
    log.causes = None
    assert all(e.cause == "planned_crossing" for e in sim.label(log, 0))


def test_snapshot_carries_conflict_cause_codes():
    sim = _sim(cause_mix={c: 0.2 for c in CAUSES})
    data = sim.generate_dataset("val", 1, 10.0, candidates="all")
    codes = np.concatenate([s.conflict_cause.numpy() for s, _ in data])
    labels = np.concatenate([l.numpy() for _, l in data])
    assert np.all((codes >= 0) == (labels > 0.5))
    assert len(set(codes[codes >= 0].tolist())) >= 2


def test_config_plumbing():
    cfg = SkyFlowConfig()
    assert cfg.make_simulator(num_uavs=10).cause_mix["planned_crossing"] == pytest.approx(0.80)
    cfg.sim.cause_mix = None
    assert cfg.make_simulator(num_uavs=10).cause_mix is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
