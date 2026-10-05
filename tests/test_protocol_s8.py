"""S8 protocol fix: input standardisation + validation-selected threshold."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyflow.models.input_norm import InputStandardizer, fit_input_norm
from skyflow.training.metrics import ConflictMetrics, MetricResult, threshold_sweep
from skyflow.training.trainer import selection_score


def _snap(x):
    return SimpleNamespace(node_features=torch.as_tensor(x, dtype=torch.float32)), torch.zeros(1)


# --------------------------------------------------------------------------- InputStandardizer
def test_standardizer_is_identity_before_fit():
    s = InputStandardizer(4)
    x = torch.randn(7, 4) * 100 + 3
    assert torch.allclose(s(x), x, atol=1e-3)
    assert not bool(s.fitted)


def test_standardizer_fit_matches_numpy_and_leaves_constant_features():
    rng = np.random.default_rng(0)
    a = rng.normal(2000.0, 500.0, size=(50, 3))
    b = rng.normal(-5.0, 0.2, size=(30, 3))
    a[:, 2] = 1.0   # constant flag column
    b[:, 2] = 1.0
    s = InputStandardizer(3).fit([_snap(a), _snap(b)])
    allrows = np.concatenate([a, b])
    assert np.allclose(s.mean.numpy(), allrows.mean(0), atol=1e-3)
    assert np.allclose(s.std.numpy()[:2], allrows.std(0)[:2], rtol=1e-4)
    assert float(s.std[2]) == 1.0            # constant feature keeps unit scale
    z = s(torch.as_tensor(allrows, dtype=torch.float32))
    assert abs(float(z[:, 0].mean())) < 1e-3 and abs(float(z[:, 0].std(unbiased=False)) - 1) < 1e-3
    assert bool(s.fitted)


def test_standardizer_state_dict_roundtrip_and_fit_switch():
    class M(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_norm = InputStandardizer(2)
            self.lin = torch.nn.Linear(2, 1)

    data = [_snap(np.array([[10.0, 0.0], [30.0, 0.0]]))]
    m = M()
    assert fit_input_norm(m, data, enabled=False) is False and not bool(m.input_norm.fitted)
    assert fit_input_norm(m, data, enabled=True) is True
    m2 = M()
    m2.load_state_dict(m.state_dict())
    assert torch.equal(m2.input_norm.mean, m.input_norm.mean) and bool(m2.input_norm.fitted)
    x = torch.tensor([[20.0, 5.0]])
    assert torch.allclose(m2.input_norm(x), m.input_norm(x))


# --------------------------------------------------------------------------- threshold sweep
def _brute_force(preds, labels, n_missed):
    best = (-1, None)
    n_pos = labels.sum() + n_missed
    for t in np.unique(preds):
        pp = preds >= t
        tp = int((pp & (labels == 1)).sum())
        fp = int((pp & (labels == 0)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / n_pos
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        if f1 > best[0] + 1e-12:
            best = (f1, t)
    return best


def test_threshold_sweep_matches_brute_force():
    rng = np.random.default_rng(1)
    for n_missed in (0, 3):
        labels = (rng.random(300) < 0.1).astype(np.float64)
        preds = np.clip(rng.normal(0.3 + 0.3 * labels, 0.2), 0, 1)
        preds = np.round(preds, 2)      # create ties
        out = threshold_sweep(preds, labels, n_missed=n_missed)
        f1_bf, thr_bf = _brute_force(preds, labels, n_missed)
        assert out["best_f1"] == pytest.approx(f1_bf, abs=1e-9)
        assert out["best_threshold"] == pytest.approx(thr_bf)
        # applying the returned threshold with ConflictMetrics reproduces best_f1
        m = ConflictMetrics(threshold=out["best_threshold"])
        m.update(torch.as_tensor(preds), torch.as_tensor(labels), latency_ms=0.0, n_missed=n_missed)
        assert m.compute().f1 == pytest.approx(out["best_f1"], abs=1e-9)


def test_threshold_sweep_auprc_toy_values():
    # perfect ranking -> AP 1 ; one positive ranked 2nd of 2 -> AP 0.5
    assert threshold_sweep(np.array([0.9, 0.8, 0.1]), np.array([1, 1, 0]))["auprc"] == pytest.approx(1.0)
    assert threshold_sweep(np.array([0.9, 0.8]), np.array([0, 1]))["auprc"] == pytest.approx(0.5)
    # missed positives lower both AP and best recall
    out = threshold_sweep(np.array([0.9, 0.1]), np.array([1, 0]), n_missed=1)
    assert out["auprc"] == pytest.approx(0.5) and out["best_cdr"] == pytest.approx(0.5)
    assert threshold_sweep(np.array([]), np.array([]))["best_f1"] == 0.0


def test_metric_result_carries_sweep_fields():
    m = ConflictMetrics(threshold=0.42)
    m.update(torch.tensor([0.1, 0.2, 0.3]), torch.tensor([0.0, 1.0, 1.0]), latency_ms=1.0)
    r = m.compute()
    assert r.f1 == 0.0 and r.threshold == 0.42                    # fixed threshold too high
    assert r.best_f1 == pytest.approx(1.0) and r.best_threshold == pytest.approx(0.2)
    assert r.auprc == pytest.approx(1.0)


# --------------------------------------------------------------------------- selection
def test_selection_score_modes():
    r = MetricResult(cdr=0.0, far=0.0, f1=0.0, precision=0.0, recall=0.0, latency_ms=0.0,
                     latency_mean_ms=0.0, num_pairs=3, num_positives=2, threshold=0.42,
                     auprc=0.9, best_f1=0.8, best_threshold=0.17, best_cdr=1.0, best_far=0.33)
    s, sel = selection_score(r, "val")
    assert s == 0.8 and sel["threshold"] == 0.17 and sel["metric"] == "best_f1@val"
    s, sel = selection_score(r, "fixed")
    assert s == 0.0 and sel["threshold"] == 0.42
    with pytest.raises(ValueError):
        selection_score(r, "bogus")
