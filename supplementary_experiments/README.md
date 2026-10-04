# Supplementary Experiments

Supplementary experiments for *SkyFlow*. Each script imports the `skyflow`
package located one level up (`../skyflow`).

## Scripts

| Script | What it measures |
|--------|------------------|
| `exp_leakage_ablation.py` | Retrains TR-GAT with the four proximity/CPA features (`n_nbr`, `d_min`, `t_cpa`, `f_avoid`; indices 19–22) removed and compares against full TR-GAT and GAT-Static under a matched protocol. |
| `exp_spatial_hashing.py` | Benchmarks brute-force pairwise CPA enumeration against a time-sampled uniform spatial index (hash grid and `scipy` KD-tree) at the 80 m proximity radius, over fleet sizes 100–2000. |

## Results

- `leakage_ablation_results.json` — per-seed CDR/FAR/F1 for full vs. no-CPA TR-GAT, plus paired t-tests.
- `spatial_hashing_results.json` — p50/p95 construction time, candidate-pair counts, and edge-set-match verification.

## Running

Dependencies are the same as the main package (`torch`, `numpy`, `scipy`, etc.).
Run from the repository root (`SkyFlowCode/`) so that the `skyflow` package is
importable, or run each script directly (paths are resolved relative to the
script location).

```bash
cd SkyFlowCode
python supplementary_experiments/exp_leakage_ablation.py
python supplementary_experiments/exp_spatial_hashing.py
```

The leakage ablation retrains models and may take several hours on a single
GPU; the spatial-hashing benchmark runs on CPU in a few minutes.
