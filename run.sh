#!/usr/bin/env bash
# SkyFlow - stage 2 of the reproduction: unified evaluation + paper artefacts (bash twin of scripts/run_s15.ps1).
#
# Run AFTER training has finished (scripts/run_main.py, see README) and while the GPU is otherwise idle:
# every latency number in the paper comes from this run.  Each stage logs to logs/s15_<stage>.log.
#
#   bash run.sh                 # all stages
#   bash run.sh robust scaling  # only the named stages
#
# Nothing in this file contains result numbers; read them from results/*.csv|json or paper/tables/*.tex.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p logs
PYTHON=${PYTHON:-python}

declare -A STAGES=(
  [eval]="scripts/eval_only.py --results_dir results/main --out_dir results/eval --latency_epochs 1000"
  [aggregate]="scripts/aggregate_main.py --results_dir results/main --eval_dir results/eval --out_prefix results/main"
  [ablation]="scripts/aggregate_ablation.py --results_dir results/main --out_csv results/ablation.csv"
  [robust]="scripts/run_robustness.py --results_dir results/main --out_dir results"
  [scaling]="scripts/run_scaling.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/scaling.csv --out_fit results/scaling_fit.json"
  [attention]="scripts/analyze_attention_aoi.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/attention_vs_aoi.csv"
  [gate]="scripts/analyze_gate.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/gate_by_cause.csv"
  [events]="scripts/eval_events.py --results_dir results/main --out_dir results --persistence 1 3"
  [nearmiss]="scripts/analyze_nearmiss.py --checkpoints results/main/TR-GAT/seed42 results/main/GAT-S/seed42 results/main/CPA-Rule/seed42 results/main/Plan-CPA/seed42 --budget_csv results/events_budget.csv --out_csv results/nearmiss.csv"
  [cpulat]="scripts/eval_only.py --results_dir results/main --methods TR-GAT --seeds 42 --device cpu --skip_graph_build --latency_epochs 200 --out_dir results/eval_cpu"
  [figures]="scripts/make_figures.py --results_dir results --out_dir paper/figs"
  [tables]="scripts/make_tables.py --results_dir results --out_dir paper/tables"
  [macros]="scripts/make_macros.py --results_dir results --out paper/numbers.tex"
)
ORDER=(eval aggregate ablation robust scaling attention gate events nearmiss cpulat figures tables macros)

if [ "$#" -gt 0 ]; then ORDER=("$@"); fi
t0=$(date +%s)
for s in "${ORDER[@]}"; do
  if [ -z "${STAGES[$s]+x}" ]; then echo "unknown stage: $s"; exit 2; fi
  echo "[$(date +%H:%M:%S)] $s: $PYTHON ${STAGES[$s]}"
  $PYTHON ${STAGES[$s]} > "logs/s15_$s.log" 2>&1
  echo "[$(date +%H:%M:%S)] $s exit=$?"
done
echo "S15 finished in $(( ($(date +%s) - t0) / 60 )) min"
