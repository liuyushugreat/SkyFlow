# SkyFlow: What Does Learning Add to Rule-Based UAV Conflict Detection under Stale Surveillance at the Edge?

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)](https://pytorch.org/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> Code, configuration and result files for the SkyFlow paper: a leakage-free, operational evaluation of
> UAV conflict detection under stale surveillance (per-scenario ADS-B latency and loss), in which three
> geometric rules and six learned detectors receive the same AoI-synchronised pair geometry and are
> compared at the level of conflict events under a common false-alert budget. TR-GAT (temporal relational
> graph attention conditioned on the age of information of every edge, recurrent state, intent-conformance
> gate) is the graph-based reference model of that comparison.

**Every number in the paper is generated from files under `results/` by the scripts in `scripts/`;
nothing is typed by hand.** `scripts/make_macros.py`, `scripts/make_tables.py` and
`scripts/make_figures.py` write the LaTeX macros, tables and figures into a local `paper/` directory,
and each generated file records the `results/` directory, git commit and hardware it came from.

**The paper itself (LaTeX sources, figures, tables, `numbers.tex`, PDF) is kept local only and is not
part of this repository** (`paper/` is git-ignored). The repository contains the code, configurations and
result files from which every number can be regenerated.

---

## What is in the artifact

| Component | Where |
|-----------|-------|
| Urban low-altitude simulator (500 UAVs, 5 km, hub-converging routes, five conflict causes, heterogeneous ADS-B latency / loss per scenario, GPS noise, non-cooperative aircraft) | `skyflow/data/` |
| Leakage-free labels: future 6-DoF trajectories integrated over 30 s, 10 m horizontal / 3 m vertical separation; inputs restricted to decision-time observables | `skyflow/data/`, `tests/test_no_leakage.py`, `tests/test_labels.py` |
| Temporal knowledge graph builder (typed nodes, five relation types, edge AoI, spatial hash candidate set) | `skyflow/data/tkg_builder.py` |
| AoI-synchronised pair geometry shared by all learned scorers and both CPA rules | `skyflow/models/`, `tests/test_pair_geometry_s8c.py`, `tests/test_s8e_sync_gate_links.py` |
| TR-GAT (AoI-conditioned relational attention, relation gating, GRU state with BPTT, intent-conformance gate) | `skyflow/models/tr_gat.py` |
| Baselines: CPA-Rule, Plan-CPA, VO, LSTM-P, Tfm-P, GAT-S, STGCN, TR-GAT-NT | `skyflow/baselines/` |
| Protocol: validation-selected thresholds, AUPRC, hard regime, paired t-tests with Bonferroni, scenario bootstrap | `skyflow/training/metrics.py`, `scripts/aggregate_main.py` |
| Event-level metrics (conflict events, lead time, false alert episodes per UAV-hour), SOC curves, EMA/hysteresis operational layer, budget-matched operating points | `skyflow/experiments/events.py`, `scripts/eval_events.py` |
| Near-miss analysis of false alerts by re-simulating the truth | `skyflow/experiments/nearmiss.py`, `scripts/analyze_nearmiss.py` |
| Robustness beyond the training link conditions, latency scaling with log-log exponents, CPU-only latency | `scripts/run_robustness.py`, `scripts/run_scaling.py`, `scripts/eval_only.py` |
| Paper artefact generators (macros, tables, figures, submission gate); the paper sources themselves are local only | `scripts/make_macros.py`, `scripts/make_tables.py`, `scripts/make_figures.py`, `scripts/check_paper.py` |

Headline numbers are **not** repeated here: read them from `results/main_summary.csv`,
`results/events_summary.csv`, `results/events_budget_summary.csv`, `results/robustness_*.csv`,
`results/scaling_fit.json` and `results/nearmiss.csv`.

---

## Installation

Prerequisites: Python 3.10+, PyTorch 2.2+ (CUDA optional; the full round was run on one RTX 4090).

```bash
git clone https://github.com/liuyushugreat/SkyFlow.git
cd SkyFlow
pip install -e ".[dev]"
python -m pytest -q          # unit tests (labels, no-leakage, AoI sync, gate, events, stats, ...)
```

Optional: `export SKYFLOW_CACHE_DIR=/path/with/space` to place the simulated-data cache outside the repo.

---

## Reproduction

The paper is produced by three stages. Stage 1 is the expensive one; stages 2-3 are evaluation only.

```bash
# 0. (optional) pre-build the simulated train/val/test cache once
python scripts/build_cache.py --config configs/default.yaml

# 1. training: 9 methods x 3 seeds, then 8 TR-GAT ablations x 3 seeds (resumable)
python scripts/run_main.py --config configs/default.yaml --results_dir results/main \
    --methods TR-GAT GAT-S STGCN LSTM-P Tfm-P TR-GAT-NT CPA-Rule Plan-CPA VO --seeds 42 123 456 --max_concurrent 2 --resume
python scripts/run_main.py --config configs/default.yaml --results_dir results/main \
    --methods abl_no_conf_gate abl_no_sync abl_no_gru abl_tbptt abl_no_plan abl_no_gating abl_bce abl_telemetry_only \
    --seeds 42 123 456 --max_concurrent 2 --resume

# 2. unified evaluation (GPU otherwise idle; every latency number comes from this run)
bash run.sh                                   # Linux/macOS: all S15 stages below
powershell -ExecutionPolicy Bypass -File scripts/run_s15.ps1    # Windows equivalent

# 3. paper artefacts are written to a local paper/ directory (git-ignored; the LaTeX sources are not in
#    this repository). With the sources present, the submission gate (fonts embedded / no Type 3, page
#    rule, no undefined macro, no smoke numbers) is:
python scripts/check_paper.py --paper_dir paper
```

Stage 2 runs, in order: `eval_only.py` (test metrics + P95 latency per checkpoint), `aggregate_main.py`,
`aggregate_ablation.py`, `run_robustness.py` (fixed latency / loss levels including levels beyond the
training mix), `run_scaling.py`, `analyze_attention_aoi.py`, `analyze_gate.py`, `eval_events.py`
(event metrics, SOC curves, budget-matched operating points selected on validation), `analyze_nearmiss.py`,
`eval_only.py --device cpu`, then `make_figures.py`, `make_tables.py`, `make_macros.py`.

Quick pipeline check without a GPU: `python scripts/run_main.py --config configs/smoke.yaml --results_dir results/_smoke --methods TR-GAT CPA-Rule --seeds 42`.

### Configuration

All settings live in `configs/default.yaml` (data, simulator, model, training, scoring). New behaviour
is always behind a switch whose default is the setting used in the paper; the older behaviour stays
reproducible (e.g. `features.leakage_free`, `features.pair_edge_features`, `features.plan_context`,
`model.use_conformance_gate`, `training.tbptt_detach`, `training.state_carry`, `sim.link_mix`).
`configs/smoke.yaml` is a tiny configuration for tests.

Methods are registered in `skyflow/experiments/methods.py` with a group: `main` (paper tables),
`ablation` (TR-GAT minus one component) and `variant` (recorded negative results that stay out of the
paper and of the Bonferroni family, e.g. `TR-GAT-SC`, GRU state carried across the windows of a
scenario; `docs/compute_plan.md` §12.3).

---

## Repository structure

```
SkyFlow/
├── run.sh                      # stage 2 (unified evaluation + paper artefacts) for bash
├── skyflow/
│   ├── data/                   # simulator, conflict causes, labels, TKG builder, cache
│   ├── models/                 # TR-GAT, temporal encoding, pair geometry, conflict head
│   ├── baselines/              # rules (CPA, Plan-CPA, VO) and learned baselines
│   ├── training/               # trainer (BPTT through the window), losses, metrics, statistics
│   └── experiments/            # event-level metrics, operational layer, near-miss analysis
├── scripts/                    # training / evaluation / aggregation / paper artefact generators
│   ├── run_main.py, run_task.py, build_cache.py
│   ├── eval_only.py, aggregate_main.py, aggregate_ablation.py
│   ├── run_robustness.py, run_scaling.py, analyze_attention_aoi.py, analyze_gate.py
│   ├── eval_events.py, analyze_nearmiss.py
│   ├── make_figures.py, make_tables.py, make_macros.py, paper_common.py
│   ├── check_paper.py          # submission gate for the compiled PDF
│   └── run_s15.ps1             # stage 2 for PowerShell
├── configs/                    # default.yaml (paper), smoke.yaml (tests)
├── tests/                      # pytest suite
├── results/                    # metrics.json per run + aggregated CSV/JSON (inputs of the paper)
└── docs/                       # compute plan and run log
# paper/ (LaTeX sources, generated numbers.tex / tables / figs, PDF) exists only locally and is git-ignored.
```

## License

This project is licensed under the Apache License 2.0.
