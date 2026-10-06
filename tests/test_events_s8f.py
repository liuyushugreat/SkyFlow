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


def _naive_tracks(tab):
    """(scenario, key) -> list of row indices in time order, split at gaps."""
    tracks, cur, prev = [], [], None
    for r in range(len(tab)):
        k = (int(tab.scenario[r]), int(tab.key[r]), int(tab.t[r]))
        if prev is not None and (k[0], k[1]) == (prev[0], prev[1]) and k[2] == prev[2] + 1:
            cur.append(r)
        else:
            if cur:
                tracks.append(cur)
            cur = [r]
        prev = k
    if cur:
        tracks.append(cur)
    return tracks


class TestOperationalLayer:
    """EMA smoothing, hysteresis and the SOC machinery (S8f, options A/D)."""

    def _random_table(self, seed=0, n_snap=12, n_pairs=15):
        rng = np.random.RandomState(seed)
        S = []
        for idx in range(n_snap):
            rows = []
            for p in range(n_pairs):
                if rng.rand() < 0.75:                       # pairs drop in and out of the candidate set
                    i, j = p, p + 20
                    if rng.rand() < 0.5:
                        i, j = j, i
                    lab = 1 if (p % 3 == 0 and 2 <= (idx % 6) <= 5) else 0
                    rows.append((i, j, rng.rand(), lab, 30 - (idx % 6) * 5 if lab else -1, 0 if lab else -1))
            S.append(_scores(idx, rows, n_uavs=40))
        return EventTable.from_scores(S, threshold=0.5, snapshots_per_scenario=6)

    def test_table_is_sorted_and_positions_follow_tracks(self):
        tab = self._random_table()
        order = np.lexsort((tab.t, tab.key, tab.scenario))
        assert (order == np.arange(len(tab))).all()
        for tr in _naive_tracks(tab):
            assert tab.position[tr].tolist() == list(range(len(tr)))

    def test_ema_matches_naive_recursion(self):
        from skyflow.experiments.events import smooth_scores
        tab = self._random_table(1)
        for alpha in (1.0, 0.5, 0.2):
            s = smooth_scores(tab, alpha)
            for tr in _naive_tracks(tab):
                acc = None
                for r in tr:
                    acc = tab.score[r] if acc is None else alpha * tab.score[r] + (1 - alpha) * acc
                    assert s[r] == pytest.approx(acc)
        with pytest.raises(ValueError):
            smooth_scores(tab, 0.0)

    def test_hysteresis_matches_naive_and_plain_threshold_is_special_case(self):
        from skyflow.experiments.events import hysteresis_alerts
        tab = self._random_table(2)
        s = tab.score
        assert (hysteresis_alerts(tab, s, 0.5, 0.5) == (s >= 0.5)).all()
        a = hysteresis_alerts(tab, s, 0.6, 0.3)
        for tr in _naive_tracks(tab):
            on = False
            for r in tr:
                on = s[r] >= 0.6 or (on and s[r] >= 0.3)
                assert bool(a[r]) == on
        assert a.sum() >= (s >= 0.6).sum()
        with pytest.raises(ValueError):
            hysteresis_alerts(tab, s, 0.3, 0.6)

    def test_soc_curve_is_monotone_and_pruning_is_exact(self):
        from skyflow.experiments.events import filtered_alerts, soc_curve, threshold_grid
        tab = self._random_table(3, n_snap=24, n_pairs=30)
        grid = threshold_grid(tab, 9)
        assert tab.threshold in grid and np.all(np.diff(grid) > 0)
        for alpha, h in ((1.0, 0.0), (0.5, 0.1)):
            pts = soc_curve(tab, grid, alpha, h, lead_s=5.0)
            cdr = [p.event_cdr for p in pts]
            # detected events shrink monotonically with the threshold (episode counts need not: a
            # higher threshold can split one long false run into two)
            assert all(x >= y - 1e-12 for x, y in zip(cdr, cdr[1:]))
            assert pts[0].false_episodes_per_uav_hour > pts[-1].false_episodes_per_uav_hour
            # same numbers without pruning (full table, explicit alert vector)
            for p in pts:
                full = event_metrics(tab, lead_s=5.0, alert=filtered_alerts(tab, p.threshold, alpha, h))
                assert (full.event_cdr, full.n_false_episodes, full.n_alert_episodes) == \
                       (p.event_cdr, p.n_false_episodes, p.n_alert_episodes)
                assert full.lead_s["median"] == pytest.approx(p.lead_s["median"], nan_ok=True)

    def test_budget_selection(self):
        from skyflow.experiments.events import select_operating_point, soc_curve, threshold_grid
        tab = self._random_table(4, n_snap=24, n_pairs=30)
        pts = soc_curve(tab, threshold_grid(tab, 9), 1.0, 0.0, lead_s=5.0)
        rates = sorted(p.false_episodes_per_uav_hour for p in pts)
        budget = rates[len(rates) // 2]
        sel = select_operating_point(pts, budget)
        assert sel is not None and sel.false_episodes_per_uav_hour <= budget
        assert sel.event_cdr == max(p.event_cdr for p in pts if p.false_episodes_per_uav_hour <= budget)
        assert select_operating_point(pts, -1.0) is None


class TestNearMiss:
    def test_rho_on_synthetic_trajectories(self):
        from skyflow.experiments.nearmiss import normalised_separation
        T, N = 50, 3
        pos = np.zeros((T, N, 3), dtype=np.float32)
        pos[:, 1, 0] = np.linspace(100.0, 0.0, T)       # UAV1 closes on UAV0 horizontally, same altitude
        pos[:, 2, 0] = 4.0                               # UAV2 4 m away but 10 m higher -> vertical keeps it apart
        pos[:, 2, 2] = 10.0
        i = np.array([0, 0]); j = np.array([1, 2])
        rho = normalised_separation(pos, 0, i, j, horizon_epochs=T - 1, h_sep=10.0, v_sep=3.0)
        assert rho[0] == pytest.approx(0.0, abs=1e-6)                     # reaches the same point
        assert rho[1] == pytest.approx(max(4.0 / 10.0, 10.0 / 3.0))
        rho_short = normalised_separation(pos, 0, i[:1], j[:1], horizon_epochs=10, h_sep=10.0, v_sep=3.0)
        assert rho_short[0] == pytest.approx(pos[10, 1, 0] / 10.0)
        # window clipped at the end of the log
        assert normalised_separation(pos, T - 1, i[:1], j[:1], 20, 10.0, 3.0)[0] == pytest.approx(0.0, abs=1e-6)

    def test_resimulated_truth_reproduces_cached_labels(self):
        """rho < 1 for every positive and >= 1 for every negative of a simulated split."""
        from skyflow.experiments.events import EventTable, SnapshotScores
        from skyflow.experiments.nearmiss import near_miss_analysis
        cfg = _tiny_cfg()
        cfg.data.num_uavs = 40
        sim = cfg.make_simulator(seed=cfg.data.sim_seed)
        data = sim.generate_dataset("test", 2, 6.0, builder=cfg.make_builder())
        logs = cfg.make_simulator(seed=cfg.data.sim_seed).simulate_logs("test", 2, 6.0)
        scores = []
        for idx, (snap, lab) in enumerate(data):
            P = snap.conflict_pairs.size(1)
            rng = np.random.RandomState(idx)
            scores.append(SnapshotScores(idx, snap.conflict_pairs.numpy(), rng.rand(P), lab.numpy(),
                                         snap.conflict_ttc.numpy(), snap.conflict_cause.numpy(), snap.num_uavs))
        sps = len(data) // 2
        tab = EventTable.from_scores(scores, 0.5, sps)
        assert tab.label.sum() > 0
        epoch_step = int(round(6.0 * cfg.data.sim_freq_hz / sps))
        res = near_miss_analysis(tab, tab.alert, logs, epoch_step, int(round(cfg.data.lookahead_seconds * cfg.data.sim_freq_hz)),
                                 cfg.data.conflict_h_sep_m, cfg.data.conflict_v_sep_m, n_negative_sample=5000)
        assert res.positive_mismatch == 0 and res.negative_mismatch == 0
        assert res.n_positive_checked == int(tab.label.sum())
        assert res.rho_false_rows.size == int((tab.alert & ~tab.label).sum())
        assert (res.rho_false_rows >= 1.0).all()
        rows = {r["group"]: r for r in res.rows()}
        assert rows["positive_check"]["frac_lt_1"] == 1.0 and rows["negative_sample"]["frac_lt_1"] == 0.0
        assert res.rho_false_episodes.size <= res.rho_false_rows.size


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
