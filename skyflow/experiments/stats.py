"""Statistics used by the aggregation scripts (S7b): power-law fit, paired
t-test with Bonferroni correction, scenario-level bootstrap."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def fit_power_law(n: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """Least-squares fit of log y = alpha * log n + log c.

    Returns alpha, c, r2 and the standard error of alpha.  For y = c * n^1.5
    alpha is 1.5 up to floating point."""
    n = np.asarray(n, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = (n > 0) & (y > 0) & np.isfinite(n) & np.isfinite(y)
    n, y = n[ok], y[ok]
    if n.size < 2:
        return {"alpha": float("nan"), "c": float("nan"), "r2": float("nan"), "alpha_se": float("nan"), "n_points": int(n.size)}
    X = np.log(n)
    Y = np.log(y)
    A = np.vstack([X, np.ones_like(X)]).T
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    alpha, logc = coef
    pred = A @ coef
    ss_res = float(np.sum((Y - pred) ** 2))
    ss_tot = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    dof = n.size - 2
    if dof > 0 and np.sum((X - X.mean()) ** 2) > 0:
        se = float(np.sqrt(ss_res / dof / np.sum((X - X.mean()) ** 2)))
    else:
        se = float("nan")
    return {"alpha": float(alpha), "c": float(np.exp(logc)), "r2": float(r2), "alpha_se": se, "n_points": int(n.size)}


def paired_ttest(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Two-sided paired t-test of a - b (paired by seed); p-value from scipy."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size < 2:
        return {"n": int(a.size), "mean_diff": float(np.mean(a - b)) if a.size else float("nan"),
                "t": float("nan"), "p": float("nan")}
    d = a - b
    sd = d.std(ddof=1)
    if sd == 0:
        t = float("inf") if d.mean() != 0 else 0.0
        p = 0.0 if d.mean() != 0 else 1.0
        return {"n": int(a.size), "mean_diff": float(d.mean()), "t": t, "p": p}
    t = float(d.mean() / (sd / np.sqrt(d.size)))
    from scipy import stats
    p = float(2 * stats.t.sf(abs(t), df=d.size - 1))
    return {"n": int(a.size), "mean_diff": float(d.mean()), "t": t, "p": p}


def bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    m = len(p_values)
    return {k: min(1.0, v * m) if np.isfinite(v) else v for k, v in p_values.items()}


def counts_to_metrics(tp: float, fp: float, fn: float) -> Tuple[float, float, float]:
    cdr = tp / max(tp + fn, 1)
    far = fp / max(tp + fp, 1)
    prec = 1.0 - far
    f1 = 2 * prec * cdr / max(prec + cdr, 1e-8)
    return float(cdr), float(far), float(f1)


def group_snapshots(per_snapshot: List[List[int]], n_groups: int) -> np.ndarray:
    """Sum per-snapshot [tp, fp, fn] into ``n_groups`` consecutive scenario
    blocks (snapshots are stored in scenario order with equal length)."""
    arr = np.asarray(per_snapshot, dtype=np.int64).reshape(-1, 3)
    if n_groups <= 0 or arr.shape[0] == 0:
        return arr.sum(axis=0, keepdims=True)
    if arr.shape[0] % n_groups != 0:
        raise ValueError(f"{arr.shape[0]} snapshots not divisible into {n_groups} scenarios")
    return arr.reshape(n_groups, -1, 3).sum(axis=1)


def bootstrap_ci(scenario_counts: List[np.ndarray], n_boot: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> Dict[str, Dict[str, float]]:
    """Bootstrap over test *scenarios*.

    ``scenario_counts``: one (S, 3) array per model seed.  Each replicate
    resamples the S scenarios with replacement (the same indices for every
    seed), computes CDR/FAR/F1 per seed and averages across seeds; the CI is
    the percentile interval of that mean."""
    rng = np.random.RandomState(seed)
    S = scenario_counts[0].shape[0]
    stacked = np.stack(scenario_counts)          # (n_seeds, S, 3)
    reps = {"cdr": [], "far": [], "f1": []}
    for _ in range(n_boot):
        idx = rng.randint(0, S, size=S)
        tot = stacked[:, idx, :].sum(axis=1)     # (n_seeds, 3)
        vals = np.array([counts_to_metrics(*row) for row in tot])  # (n_seeds, 3)
        m = vals.mean(axis=0)
        reps["cdr"].append(m[0]); reps["far"].append(m[1]); reps["f1"].append(m[2])
    out = {}
    for k, v in reps.items():
        v = np.asarray(v)
        out[k] = {"lo": float(np.percentile(v, 100 * alpha / 2)), "hi": float(np.percentile(v, 100 * (1 - alpha / 2))),
                  "boot_mean": float(v.mean())}
    return out
