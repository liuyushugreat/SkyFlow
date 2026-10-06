"""S8f: event-level (operational) metrics.

  * conflict events are maximal runs of consecutive positive snapshots of an
    unordered pair inside one scenario (hand-built table, exact counts);
  * lead time is the ttc at the first alerted snapshot, timely = lead >= L
    among events where that was possible;
  * alert episodes are runs of alerted snapshots, true iff they overlap a
    positive; false episodes are normalised per UAV-hour;
  * a detector that alerts exactly on the labels scores 1.0 everywhere, one
    that never alerts scores 0 / NaN; pair order (i,j)/(j,i) does not matter;
  * the per-snapshot scorer reproduces SkyFlowTrainer.evaluate's counts and
    the whole pipeline runs on a TR-GAT, a learned and a rule method.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from skyflow.experiments.events import (EventTable, SnapshotScores, event_metrics, iter_scores,
                                        window_index_groups)


def _scores(index, rows, n_uavs=10):
    """rows: list of (i, j, score, label, ttc, cause)."""
    if not rows:
        z = np.zeros(0)
        return SnapshotScores(index, np.zeros((2, 0), np.int64), z, z, z, np.zeros(0, np.int8), n_uavs)
    a = np.array(rows, dtype=float)
    return SnapshotScores(index, a[:, :2].T.astype(np.int64), a[:, 2], a[:, 3], a[:, 4],
                          a[:, 5].astype(np.int8), n_uavs)


def _toy():
    """2 scenarios x 6 snapshots, 10 UAVs.
    scenario 0: pair (1,2) positive t=0..3 (ttc 12,11,10,9), alerted t=2,3 -> lead 10, timely (L=10)
                pair (3,4) positive t=4..5 (ttc 5,4), never alerted
                pair (5,6) negative, alerted t=0..1 (false episode, 2 s) and t=3 (false episode, 1 s)
    scenario 1: pair (2,1) [reversed order] positive t=1..2 (ttc 3,2), alerted t=0 (negative, ttc -1) and t=1
                -> detected, lead 3, NOT timely-possible (max ttc 3 < 10); the alert run t=0..1 is ONE true episode
    """
    S = []
    S.append(_scores(0, [(1, 2, 0.1, 1, 12, 0), (5, 6, 0.9, 0, -1, -1)]))
    S.append(_scores(1, [(1, 2, 0.2, 1, 11, 0), (5, 6, 0.8, 0, -1, -1)]))
    S.append(_scores(2, [(1, 2, 0.9, 1, 10, 0), (5, 6, 0.1, 0, -1, -1)]))
    S.append(_scores(3, [(1, 2, 0.7, 1, 9, 0), (5, 6, 0.6, 0, -1, -1)]))
    S.append(_scores(4, [(3, 4, 0.1, 1, 5, 2)]))
    S.append(_scores(5, [(3, 4, 0.2, 1, 4, 2)]))
    S.append(_scores(6, [(1, 2, 0.9, 0, -1, -1)]))
    S.append(_scores(7, [(2, 1, 0.9, 1, 3, 1)]))
    S.append(_scores(8, [(2, 1, 0.1, 1, 2, 1)]))
    S.append(_scores(9, []))
    S.append(_scores(10, []))
    S.append(_scores(11, []))
    return S


class TestEventLogic:
    def test_hand_built_table(self):
        tab = EventTable.from_scores(_toy(), threshold=0.5, snapshots_per_scenario=6)
        assert tab.n_scenarios == 2 and tab.num_uavs == 10
        r = event_metrics(tab, lead_s=10.0, snapshot_dt_s=1.0)
        assert r.n_events == 3 and r.n_detected == 2
        assert r.event_cdr == pytest.approx(2 / 3)
        # timely possible: (1,2) max ttc 12 and NOT (3,4) 5 / (2,1) 3
        assert r.n_timely_possible == 1 and r.n_timely == 1 and r.timely_cdr == 1.0
        assert sorted(r.lead_s_list.tolist()) == [3.0, 10.0]
        assert r.lead_s["median"] == pytest.approx(6.5)
        # lead fraction: 10/12 and 3/3
        assert r.lead_frac == pytest.approx((10 / 12 + 1.0) / 2)
        # alert episodes: (1,2)@t2-3 true, (5,6)@t0-1 false, (5,6)@t3 false, scen1 (1,2)@t0-1 true
        assert r.n_alert_episodes == 4 and r.n_true_episodes == 2 and r.n_false_episodes == 2
        assert r.episode_precision == pytest.approx(0.5)
        assert r.uav_hours == pytest.approx(10 * 2 * 6 / 3600)
        assert r.false_episodes_per_uav_hour == pytest.approx(2 / r.uav_hours)
        assert r.false_episode_duration_s["mean"] == pytest.approx(1.5)
        assert r.per_cause["planned_crossing"]["n_events"] == 1 and r.per_cause["planned_crossing"]["event_cdr"] == 1.0
        assert r.per_cause["nonconforming"]["event_cdr"] == 0.0
        assert r.per_cause["wind_deviation"]["lead_median_s"] == 3.0

    def test_oracle_and_silent_detectors(self):
        S = _toy()
        oracle = [SnapshotScores(s.index, s.pairs, s.labels.copy(), s.labels, s.ttc, s.cause, s.num_uavs) for s in S]
        r = event_metrics(EventTable.from_scores(oracle, 0.5, 6), lead_s=10.0)
        assert r.event_cdr == 1.0 and r.timely_cdr == 1.0 and r.n_false_episodes == 0
        assert r.episode_precision == 1.0 and r.false_episodes_per_uav_hour == 0.0
        assert sorted(r.lead_s_list.tolist()) == [3.0, 5.0, 12.0]
        silent = [SnapshotScores(s.index, s.pairs, np.zeros_like(s.scores), s.labels, s.ttc, s.cause, s.num_uavs) for s in S]
        r = event_metrics(EventTable.from_scores(silent, 0.5, 6), lead_s=10.0)
        assert r.event_cdr == 0.0 and r.n_alert_episodes == 0 and math.isnan(r.episode_precision)

    def test_gap_splits_events_and_scenario_boundary_splits_runs(self):
        # same pair positive at t=0,1 and t=3,4 -> two events; alerted continuously t=0..4 -> one TRUE episode
        S = [_scores(0, [(1, 2, 1, 1, 8, 0)]), _scores(1, [(1, 2, 1, 1, 7, 0)]), _scores(2, [(1, 2, 1, 0, -1, -1)]),
             _scores(3, [(1, 2, 1, 1, 9, 0)]), _scores(4, [(1, 2, 1, 1, 8, 0)]),
             # next scenario starts at index 5: positive again, must not merge with the run above
             _scores(5, [(1, 2, 1, 1, 6, 0)])]
        r = event_metrics(EventTable.from_scores(S, 0.5, 5), lead_s=10.0)
        assert r.n_events == 3 and r.n_detected == 3
        assert r.n_alert_episodes == 2 and r.n_true_episodes == 2

    def test_persistence_filter(self):
        from skyflow.experiments.events import persistent_alerts
        tab = EventTable.from_scores(_toy(), threshold=0.5, snapshots_per_scenario=6)
        assert (persistent_alerts(tab, 1) == tab.alert).all()
        a2 = persistent_alerts(tab, 2)
        # alert runs: (1,2)@t2-3 -> only t3 survives; (5,6)@t0-1 -> t1; (5,6)@t3 -> dropped; scen1 (1,2)@t0-1 -> t1
        assert a2.sum() == 3 and a2.sum() < tab.alert.sum()
        r2 = event_metrics(tab, lead_s=10.0, persistence=2)
        assert r2.n_events == 3 and r2.n_detected == 2            # both detected events still detected
        assert sorted(r2.lead_s_list.tolist()) == [3.0, 9.0]       # lead of (1,2) shrinks by one snapshot
        assert r2.n_false_episodes == 1 and r2.n_true_episodes == 2
        assert r2.timely_cdr == 0.0                               # 9 s < 10 s
        r9 = event_metrics(tab, lead_s=10.0, persistence=9)
        assert r9.n_alert_episodes == 0 and r9.n_detected == 0

    def test_window_groups_match_trainer(self):
        from skyflow.config import SkyFlowConfig
        from skyflow.training.trainer import SkyFlowTrainer
        tr = SkyFlowTrainer(SkyFlowConfig(), device=torch.device("cpu"))
        for n, K in ((12, 4), (13, 4), (3, 5), (0, 3)):
            data = list(range(n))
            want = [list(w) for w in tr._group_into_windows(data, K)]
            assert window_index_groups(n, K) == want


def _load_script(name):
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        name, pathlib.Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class TestRobustnessConditions:
    """The sweep must impose a homogeneous level on every scenario (link_mix off)
    and flag levels beyond the training mix."""

    def test_condition_overrides_link_mix_and_changes_cache_key(self):
        from skyflow.config import SkyFlowConfig
        from skyflow.data.cache import cache_key
        rb = _load_script("run_robustness")
        base = SkyFlowConfig()
        assert base.sim.link_mix                                  # default = heterogeneous mix
        c3 = rb.condition_cfg(base, "latency", 3.0, 0.4)
        assert c3.sim.link_mix is None and c3.sim.packet_loss == 0.0
        assert c3.sim.adsb_latency_s == pytest.approx([1.8, 4.2])
        sim = c3.make_simulator(seed=1)
        logs = sim.simulate_logs("test", 2, 1.0)
        for lg in logs:                                           # every scenario sees the fixed level
            assert sim.scenario_obs_params(lg).latency_range() == pytest.approx((1.8, 4.2))
            assert sim.scenario_obs_params(lg).packet_loss == 0.0
        keys = {cache_key(rb.condition_cfg(base, "loss", v, 0.4), "test") for v in (0.0, 0.2, 0.5)}
        assert len(keys) == 3 and cache_key(base, "test") not in keys

    def test_in_train_range_flag(self):
        from skyflow.config import SkyFlowConfig
        rb = _load_script("run_robustness")
        base = SkyFlowConfig()
        hi_lat = base.sim.link_mix["latency_hi_s"][1]
        hi_loss = base.sim.link_mix["packet_loss"][1]
        inside = rb.condition_cfg(base, "latency", hi_lat / 1.4, 0.4)
        outside = rb.condition_cfg(base, "latency", hi_lat / 1.4 + 0.5, 0.4)
        assert rb.in_train_range(base, "latency", inside) == 1
        assert rb.in_train_range(base, "latency", outside) == 0
        assert rb.in_train_range(base, "loss", rb.condition_cfg(base, "loss", hi_loss, 0.4)) == 1
        assert rb.in_train_range(base, "loss", rb.condition_cfg(base, "loss", hi_loss + 0.1, 0.4)) == 0
        assert any(not rb.in_train_range(base, "latency", rb.condition_cfg(base, "latency", v, 0.4))
                   for v in rb.DEFAULT_LATENCIES)
        assert any(not rb.in_train_range(base, "loss", rb.condition_cfg(base, "loss", v, 0.4))
                   for v in rb.DEFAULT_LOSSES)
        # legacy: no mix -> compare with the fixed training condition
        base.sim.link_mix = None
        base.sim.adsb_latency_s = [0.5, 1.2]
        assert rb.in_train_range(base, "latency", rb.condition_cfg(base, "latency", 0.5, 0.4)) == 1
        assert rb.in_train_range(base, "latency", rb.condition_cfg(base, "latency", 2.0, 0.4)) == 0


def _tiny_cfg():
    from skyflow.config import SkyFlowConfig
    cfg = SkyFlowConfig()
    cfg.model.num_layers, cfg.model.embed_dim, cfg.model.num_heads = 1, 16, 2
    cfg.model.temporal_dim, cfg.model.recurrent_dim = 8, 8
    cfg.data.num_uavs, cfg.data.grid_size_m, cfg.data.observation_window = 20, 600.0, 3
    cfg.data.num_sectors, cfg.data.num_weather_cells, cfg.data.num_restricted_zones = 4, 4, 2
    cfg.data.test_scenarios, cfg.data.scenario_duration_s = 2, 6.0
    cfg.training.epochs, cfg.training.min_epochs, cfg.training.early_stopping_patience = 1, 1, 0
    return cfg


class TestPipeline:
    def test_trgat_scores_match_evaluate_and_events_run(self, tmp_path):
        from skyflow.experiments.loader import LoadedMethod
        from skyflow.experiments.events import evaluate_events, method_threshold
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = _tiny_cfg()
        sim = cfg.make_simulator(seed=5)
        data = sim.generate_dataset("test", 2, 6.0, builder=cfg.make_builder())
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
        tr.train(data, data, seed=1, output_dir=str(tmp_path), max_epochs=1)
        lm = LoadedMethod("TR-GAT", "trgat", 1, cfg, tmp_path, tr.evaluate, trainer=tr)
        ref = tr.evaluate(data)
        scores = list(iter_scores(lm, data))
        assert [s.index for s in scores] == list(range(len(data)))
        thr = method_threshold(lm)
        tp = fp = fn = 0
        for s in scores:
            a, y = s.scores >= thr, s.labels >= 0.5
            tp += int((a & y).sum()); fp += int((a & ~y).sum()); fn += int((~a & y).sum())
        want = np.asarray(ref.per_snapshot).reshape(-1, 3).sum(axis=0)
        assert (tp, fp, fn) == tuple(int(v) for v in want)
        sps = len(data) // cfg.data.test_scenarios
        r, r3 = evaluate_events(lm, data, snapshots_per_scenario=sps, lead_s=2.0, snapshot_dt_s=6.0 / sps, persistence=(1, 3))
        assert r3.n_events == r.n_events and r3.n_alert_episodes <= r.n_alert_episodes
        assert r.n_events >= 0 and r.uav_hours > 0
        row = r.to_row()
        assert set(row) >= {"event_cdr", "timely_cdr", "false_episodes_per_uav_hour", "episode_precision"}

    def test_rule_and_learned_baselines_run(self):
        from skyflow.baselines.registry import get_baseline
        from skyflow.experiments.loader import LoadedMethod
        from skyflow.experiments.events import evaluate_events
        cfg = _tiny_cfg()
        sim = cfg.make_simulator(seed=5)
        data = sim.generate_dataset("test", 2, 6.0, builder=cfg.make_builder())
        dev = torch.device("cpu")
        rule = get_baseline("CPA-Rule", cfg, dev)
        rule.fit(data)
        lm = LoadedMethod("CPA-Rule", "rule", 42, cfg, None, None, model=rule)
        (r,) = evaluate_events(lm, data, snapshots_per_scenario=len(data) // 2, lead_s=2.0)
        assert r.n_events >= 0 and 0.0 <= (r.event_cdr if r.n_events else 0.0) <= 1.0
        gat = get_baseline("GAT-S", cfg, dev)
        gat.threshold = 0.5
        lm = LoadedMethod("GAT-S", "learned", 42, cfg, None, None, model=gat)
        (r2,) = evaluate_events(lm, data, snapshots_per_scenario=len(data) // 2, lead_s=2.0)
        assert r2.n_events == r.n_events      # events depend on labels only, not on the method
