"""S7b: statistics helpers used by the aggregation scripts."""

import numpy as np
import pytest

from skyflow.experiments.stats import (bonferroni, bootstrap_ci, counts_to_metrics, fit_power_law,
                                       group_snapshots, paired_ttest)
from skyflow.training.metrics import ConflictMetrics
import torch


def test_fit_power_law_recovers_exponent():
    n = np.array([100, 250, 500, 1000, 2000])
    y = 3.0 * n ** 1.5
    fit = fit_power_law(n, y)
    assert fit["alpha"] == pytest.approx(1.5, abs=1e-9)
    assert fit["c"] == pytest.approx(3.0, rel=1e-9)
    assert fit["r2"] == pytest.approx(1.0, abs=1e-12)
    assert fit["n_points"] == 5


def test_fit_power_law_with_noise_and_bad_points():
    rng = np.random.RandomState(0)
    n = np.array([100, 250, 500, 1000, 2000, 4000], dtype=float)
    y = 0.5 * n ** 1.2 * np.exp(rng.normal(0, 0.02, n.size))
    y[-1] = np.nan                                  # an OOM row
    fit = fit_power_law(n, y)
    assert fit["n_points"] == 5
    assert fit["alpha"] == pytest.approx(1.2, abs=0.05)
    assert fit["alpha_se"] < 0.05


def test_fit_power_law_too_few_points():
    fit = fit_power_law([100], [1.0])
    assert np.isnan(fit["alpha"])


def test_paired_ttest_and_bonferroni():
    a = [0.80, 0.82, 0.79, 0.81, 0.83]
    b = [0.70, 0.73, 0.69, 0.72, 0.71]
    t = paired_ttest(a, b)
    assert t["n"] == 5 and t["mean_diff"] == pytest.approx(0.1)
    assert t["p"] < 1e-3
    same = paired_ttest(a, a)
    assert same["p"] == 1.0
    adj = bonferroni({"x": 0.01, "y": 0.4, "z": float("nan")})
    assert adj["x"] == pytest.approx(0.03) and adj["y"] == 1.0 and np.isnan(adj["z"])


def test_counts_group_and_bootstrap():
    # 2 seeds, 4 scenarios x 3 snapshots each
    per_snap = [[5, 1, 2]] * 12
    g = group_snapshots(per_snap, 4)
    assert g.shape == (4, 3) and g[0].tolist() == [15, 3, 6]
    with pytest.raises(ValueError):
        group_snapshots(per_snap, 5)
    cdr, far, f1 = counts_to_metrics(15, 3, 6)
    assert cdr == pytest.approx(15 / 21) and far == pytest.approx(3 / 18)
    ci = bootstrap_ci([g, g], n_boot=200, seed=1)
    # identical scenarios -> degenerate CI equal to the point estimate
    assert ci["cdr"]["lo"] == pytest.approx(cdr) and ci["cdr"]["hi"] == pytest.approx(cdr)
    g2 = g.copy()
    g2[0] = [0, 0, 20]                              # one bad scenario -> CI widens
    ci2 = bootstrap_ci([g2], n_boot=300, seed=1)
    assert ci2["cdr"]["lo"] < ci2["cdr"]["hi"]
    assert ci2["cdr"]["lo"] < cdr


def test_conflict_metrics_per_snapshot_matches_totals():
    m = ConflictMetrics(threshold=0.5)
    m.update(torch.tensor([0.9, 0.1, 0.8, 0.7]), torch.tensor([1.0, 1.0, 0.0, 1.0]), n_missed=2)
    m.add_missed(3)                                   # snapshot without candidate pairs
    m.add_missed(0)                                   # snapshot without positives
    m.update(torch.tensor([0.2]), torch.tensor([0.0]))
    res = m.compute()
    ps = np.array(res.per_snapshot)
    assert ps.shape == (4, 3)
    assert ps.tolist() == [[2, 1, 3], [0, 0, 3], [0, 0, 0], [0, 0, 0]]
    tp, fp, fn = ps.sum(axis=0)
    assert res.num_missed_positives == 5
    assert res.cdr == pytest.approx(tp / (tp + fn))
    assert res.far == pytest.approx(fp / (tp + fp))
