"""S8g: ``training.state_carry`` - GRU state carried across the windows of a
scenario ("scenario") or reset at every window ("window", the S8e behaviour).

Checks
  * window_sequences(): identical window boundaries in both modes; sequences
    follow scenario boundaries; invalid mode rejected;
  * the trainer resets the recurrent state once per window ("window") and once
    per scenario ("scenario") in evaluate() and in the shuffled training
    schedule, which keeps the windows of a scenario in time order;
  * one training epoch in "scenario" mode runs end to end;
  * events.iter_scores follows the same schedule and scores every snapshot once;
  * the TR-GAT-SC method sets the switch, TR-GAT keeps the default "window".
"""

import tempfile

import pytest
import torch

from skyflow.config import SkyFlowConfig
from skyflow.training.windows import scenario_length, window_index_groups, window_sequences


def _flat(seqs):
    return [g for s in seqs for g in s]


class TestWindowSequences:
    def test_window_mode_one_window_per_sequence(self):
        seqs = window_sequences(30, 10, 10, "window")
        assert seqs == [[g] for g in window_index_groups(30, 10)]

    def test_scenario_mode_groups_by_scenario_and_keeps_boundaries(self):
        seqs = window_sequences(40, 10, 20, "scenario")
        assert [len(s) for s in seqs] == [2, 2]
        assert _flat(seqs) == window_index_groups(40, 10)
        assert seqs[1][0][0] == 20            # second scenario starts a new sequence

    def test_tail_window_joins_its_scenario(self):
        groups = window_index_groups(25, 10)
        assert groups[-1] == list(range(15, 25))
        seqs = window_sequences(25, 10, 25, "scenario")
        assert len(seqs) == 1 and seqs[0] == groups

    def test_invalid_mode(self):
        with pytest.raises(ValueError):
            window_sequences(30, 10, 10, "flight")

    def test_scenario_length_from_config(self):
        cfg = SkyFlowConfig()
        assert scenario_length(cfg) == 60                      # 60 s at 1 Hz snapshots
        cfg.data.scenario_duration_s = 30.0
        assert scenario_length(cfg) == 30


def _tiny_cfg(state_carry):
    cfg = SkyFlowConfig()
    cfg.model.num_layers, cfg.model.embed_dim, cfg.model.num_heads = 1, 16, 2
    cfg.model.temporal_dim, cfg.model.recurrent_dim = 8, 8
    cfg.data.num_uavs, cfg.data.grid_size_m, cfg.data.observation_window = 12, 600.0, 3
    cfg.data.num_sectors, cfg.data.num_weather_cells, cfg.data.num_restricted_zones = 4, 4, 2
    cfg.data.scenario_duration_s = 30.0           # 30 snapshots per scenario (1 Hz)
    cfg.training.epochs, cfg.training.min_epochs, cfg.training.early_stopping_patience = 1, 1, 0
    cfg.training.state_carry = state_carry
    return cfg


def _data(cfg, n_scenarios=2):
    sim = cfg.make_simulator(seed=5)
    return sim.generate_dataset("train", n_scenarios, cfg.data.scenario_duration_s, builder=cfg.make_builder())


class _ResetCounter:
    """Wraps model.forward and counts calls with recurrent_state=None."""

    def __init__(self, model):
        self.model, self.resets, self.calls = model, 0, 0
        self._orig = model.forward

    def __enter__(self):
        def fwd(*a, **k):
            self.calls += 1
            if k.get("recurrent_state", None) is None and (len(a) < 4 or a[3] is None):
                self.resets += 1
            return self._orig(*a, **k)
        self.model.forward = fwd
        return self

    def __exit__(self, *exc):
        self.model.forward = self._orig


class TestTrainerStateCarry:
    @pytest.mark.parametrize("mode,expected_resets", [("window", 20), ("scenario", 2)])
    def test_evaluate_resets_once_per_window_or_scenario(self, mode, expected_resets):
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = _tiny_cfg(mode)
        data = _data(cfg)
        assert len(data) == 60
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
        with _ResetCounter(tr.model) as rc:
            tr.evaluate(data, threshold=0.5)
        assert rc.calls == 60 and rc.resets == expected_resets

    def test_shuffled_schedule_keeps_scenario_windows_in_order(self):
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = _tiny_cfg("scenario")
        data = _data(cfg)
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu"))
        pos = {id(s): i for i, (s, _) in enumerate(data)}
        schedule = tr._window_schedule(data, cfg.data.observation_window, shuffle=True)
        assert len(schedule) == 20 and sum(r for _, r in schedule) == 2
        prev = None
        for window, reset in schedule:
            first = pos[id(window[0][0])]
            if reset:
                assert first % 30 == 0
            else:
                assert first == prev + 3                       # consecutive window of the same scenario
            prev = first
        # "window" mode: every window resets, same window set
        cfg.training.state_carry = "window"
        sched_w = SkyFlowTrainer(cfg, device=torch.device("cpu"))._window_schedule(data, 3, shuffle=True)
        assert len(sched_w) == 20 and all(r for _, r in sched_w)

    def test_one_epoch_in_scenario_mode_runs(self):
        from skyflow.training.trainer import SkyFlowTrainer
        cfg = _tiny_cfg("scenario")
        data = _data(cfg, n_scenarios=1)
        tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
        with tempfile.TemporaryDirectory() as d, _ResetCounter(tr.model) as rc:
            info = tr.train(data, data, seed=1, output_dir=d, max_epochs=1)
        assert info["epochs_run"] == 1
        # 1 training pass + 1 validation pass over one scenario -> 2 resets, 60 calls
        assert rc.resets == 2 and rc.calls == 60

    def test_iter_scores_follows_the_schedule(self):
        from skyflow.experiments.events import iter_scores
        from skyflow.experiments.loader import LoadedMethod
        from skyflow.training.trainer import SkyFlowTrainer
        for mode, expected in (("window", 20), ("scenario", 2)):
            cfg = _tiny_cfg(mode)
            data = _data(cfg)
            tr = SkyFlowTrainer(cfg, device=torch.device("cpu")); tr.build_model()
            lm = LoadedMethod("TR-GAT", "trgat", 1, cfg, None, tr.evaluate, trainer=tr)
            with _ResetCounter(tr.model) as rc:
                idx = [s.index for s in iter_scores(lm, data)]
            assert idx == list(range(60)) and rc.resets == expected


def test_method_switch():
    from skyflow.experiments.methods import method_config, METHODS
    assert method_config("TR-GAT", SkyFlowConfig()).training.state_carry == "window"
    assert method_config("TR-GAT-SC", SkyFlowConfig()).training.state_carry == "scenario"
    assert METHODS["TR-GAT-SC"].group == "variant"             # recorded variant, not a paper method


def test_eval_events_skips_recorded_variants_by_default(tmp_path):
    """TR-GAT-SC results stay on disk but do not enter the event-level evaluation (Bonferroni family,
    per-method macros) unless named explicitly."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "eval_events", pathlib.Path(__file__).resolve().parents[1] / "scripts" / "eval_events.py")
    ev = importlib.util.module_from_spec(spec); spec.loader.exec_module(ev)
    for m in ("TR-GAT", "TR-GAT-SC", "abl_no_gru", "CPA-Rule"):
        for s in (42, 123):
            d = tmp_path / m / f"seed{s}"
            d.mkdir(parents=True)
            (d / "DONE").touch()
    default = {(m, s) for m, s, _ in ev.select_tasks(tmp_path)}
    assert default == {(m, s) for m in ("TR-GAT", "abl_no_gru", "CPA-Rule") for s in (42, 123)}
    explicit = {(m, s) for m, s, _ in ev.select_tasks(tmp_path, methods=["TR-GAT-SC"], seeds=[42])}
    assert explicit == {("TR-GAT-SC", 42)}
