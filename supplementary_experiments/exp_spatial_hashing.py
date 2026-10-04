#!/usr/bin/env python3
"""Spatial-hashing graph construction benchmark.

Compares O(N^2) brute-force pairwise CPA enumeration (Algorithm 1 as published)
against uniform spatial hashing at the 80 m proximity radius, which reduces the
candidate set to O(N * k_bar). Measures wall-clock construction time at fleet
sizes 100..2000 and reports candidate-pair counts.

Both implementations are vectorized NumPy for a fair comparison.
"""

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# Resolve the SkyFlow package root relative to this file.
SKYFLOW_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SKYFLOW_ROOT))

from skyflow.data.urbanair500 import UrbanAir500

CPA_H = 80.0        # approach proximity radius (m), Alg. 1
CPA_V = 15.0
LOOKAHEAD = 60.0
# Max horizontal closing speed between two small UAVs (2 x 22 m/s cruise)
V_MAX_REL = 44.0
FLEET_SIZES = [100, 200, 300, 400, 500, 1000, 2000]
N_EPOCH_SAMPLES = 50
OUT = Path(__file__).parent / "spatial_hashing_results.json"


def cpa_filter(pos, vel, pairs_i, pairs_j):
    """Vectorized CPA computation for candidate pairs -> approach-edge mask."""
    if len(pairs_i) == 0:
        return np.zeros(0, dtype=bool)
    dp = pos[pairs_j] - pos[pairs_i]
    dv = vel[pairs_j] - vel[pairs_i]
    v_dist = np.abs(dp[:, 2])
    dvdv = np.einsum("ij,ij->i", dv, dv)
    dpdv = np.einsum("ij,ij->i", dp, dv)
    t_cpa = np.where(dvdv > 1e-8, np.clip(-dpdv / np.maximum(dvdv, 1e-8), 0, LOOKAHEAD), 0.0)
    cpa = dp + dv * t_cpa[:, None]
    cpa_h = np.hypot(cpa[:, 0], cpa[:, 1])
    h_now = np.hypot(dp[:, 0], dp[:, 1])
    closing = dpdv < 0
    eff_h = np.where(closing, cpa_h, h_now)
    return (eff_h < CPA_H) & (v_dist < CPA_V)


def brute_force(pos, vel):
    n = len(pos)
    ii, jj = np.triu_indices(n, k=1)
    mask = cpa_filter(pos, vel, ii, jj)
    return int(len(ii)), int(mask.sum())


N_TIME_SAMPLES = 9  # look-ahead sampled every 7.5 s


def spatial_hash(pos, vel):
    """Time-sampled uniform spatial hashing over the look-ahead horizon.

    The look-ahead [0, 60] s is sampled at 9 instants (step 7.5 s). At each
    sample the linearly extrapolated positions are hashed into a uniform grid
    with cell size c = CPA_H + V_MAX_REL * step/2 = 245 m. If a pair's CPA
    distance is < 80 m at any t* in the horizon, their separation at the
    nearest sample instant is < c, so they share the same or an adjacent
    cell: the candidate set is a provable superset of all approach edges.
    Candidates are deduplicated across samples, then passed through the same
    exact CPA filter as brute force.
    """
    n = len(pos)
    step = LOOKAHEAD / (N_TIME_SAMPLES - 1)
    cell = CPA_H + V_MAX_REL * step / 2.0

    cand_keys = set()
    for s in range(N_TIME_SAMPLES):
        p = pos[:, :2] + vel[:, :2] * (s * step)
        keys = np.floor(p / cell).astype(np.int64)
        buckets = defaultdict(list)
        for idx in range(n):
            buckets[(keys[idx, 0], keys[idx, 1])].append(idx)

        for (cx, cy), members in buckets.items():
            m = np.array(members)
            if len(m) > 1:
                ii, jj = np.triu_indices(len(m), k=1)
                for a, b in zip(m[ii], m[jj]):
                    cand_keys.add(a * n + b if a < b else b * n + a)
            for dx, dy in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                other = buckets.get((cx + dx, cy + dy))
                if other:
                    for a in members:
                        for b in other:
                            cand_keys.add(a * n + b if a < b else b * n + a)

    if cand_keys:
        arr = np.fromiter(cand_keys, dtype=np.int64, count=len(cand_keys))
        ci, cj = arr // n, arr % n
    else:
        ci = np.zeros(0, dtype=np.int64); cj = np.zeros(0, dtype=np.int64)
    mask = cpa_filter(pos, vel, ci, cj)
    return int(len(ci)), int(mask.sum())


def spatial_index_kdtree(pos, vel):
    """Same time-sampled candidate generation, but with a C-implemented
    2-d tree (scipy.spatial.cKDTree) instead of the Python dict grid.
    Exactness argument identical to spatial_hash()."""
    from scipy.spatial import cKDTree

    n = len(pos)
    step = LOOKAHEAD / (N_TIME_SAMPLES - 1)
    radius = CPA_H + V_MAX_REL * step / 2.0

    keys = []
    for s in range(N_TIME_SAMPLES):
        p = pos[:, :2] + vel[:, :2] * (s * step)
        tree = cKDTree(p)
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        if len(pairs):
            keys.append(pairs[:, 0] * n + pairs[:, 1])
    if keys:
        arr = np.unique(np.concatenate(keys))
        ci, cj = arr // n, arr % n
    else:
        ci = np.zeros(0, dtype=np.int64); cj = np.zeros(0, dtype=np.int64)
    mask = cpa_filter(pos, vel, ci, cj)
    return int(len(ci)), int(mask.sum())


def bench(fn, states):
    times, cands, edges = [], [], []
    for pos, vel in states:
        t0 = time.perf_counter()
        n_cand, n_edge = fn(pos, vel)
        times.append((time.perf_counter() - t0) * 1000.0)
        cands.append(n_cand); edges.append(n_edge)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "mean_candidates": float(np.mean(cands)),
        "mean_edges": float(np.mean(edges)),
    }


def collect_states(n_uav, n_samples):
    sim = UrbanAir500(num_uavs=n_uav, seed=42)
    # Skip the simulator's O(N^2) pure-Python ground-truth labeling loop:
    # this benchmark only needs kinematic states, not labels.
    sim._detect_ground_truth_conflicts = lambda *a, **k: []
    states = []
    plans = sim.generate_flight_plans(n_uav)
    for epoch_idx, (state, _) in enumerate(sim.simulate_scenario(60.0, plans)):
        if epoch_idx % 10 != 0:
            continue
        states.append((state.uav_positions.copy(), state.uav_velocities.copy()))
        if len(states) >= n_samples:
            break
    return states


def main():
    results = {}
    for n in FLEET_SIZES:
        print(f"\nFleet size {n}: generating states...", flush=True)
        t0 = time.perf_counter()
        states = collect_states(n, N_EPOCH_SAMPLES)
        print(f"  {len(states)} states in {time.perf_counter()-t0:.0f}s", flush=True)

        bf = bench(brute_force, states)
        sh = bench(spatial_index_kdtree, states)
        # sanity: same number of approach edges recovered
        match = abs(bf["mean_edges"] - sh["mean_edges"]) < 1e-6
        results[n] = {"brute_force": bf, "spatial_index": sh, "edges_match": bool(match)}
        print(f"  brute-force : p95 {bf['p95_ms']:8.2f} ms | candidates {bf['mean_candidates']:>12.0f} | edges {bf['mean_edges']:.1f}", flush=True)
        print(f"  spatial-idx : p95 {sh['p95_ms']:8.2f} ms | candidates {sh['mean_candidates']:>12.0f} | edges {sh['mean_edges']:.1f} | match={match}", flush=True)

        with open(OUT, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\nSaved {OUT}", flush=True)


if __name__ == "__main__":
    main()
