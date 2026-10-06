# S15: unified evaluation on the local 4090 after ALL training has finished (GPU must be otherwise idle).
# Every latency number in the paper comes from this run.  Re-runnable; each stage logs to logs/s15_*.log.
#   powershell -ExecutionPolicy Bypass -File scripts/run_s15.ps1
$ErrorActionPreference = "Continue"
Set-Location (Split-Path $PSScriptRoot -Parent)
New-Item -ItemType Directory -Force logs | Out-Null
$t0 = Get-Date
function Stage($name, $cmd) {
    Write-Host ("[{0}] {1}: {2}" -f (Get-Date -Format "HH:mm:ss"), $name, $cmd)
    Invoke-Expression "$cmd > logs/s15_$name.log 2>&1"
    Write-Host ("[{0}] {1} exit={2}" -f (Get-Date -Format "HH:mm:ss"), $name, $LASTEXITCODE)
}
# 1. test metrics + P95 latency for every checkpoint (3-stage timing, same host)
Stage eval      "python scripts/eval_only.py --results_dir results/main --out_dir results/eval --latency_epochs 1000"
# 2. aggregation (main table, per-regime, paired t-tests + Bonferroni, scenario bootstrap)
Stage aggregate "python scripts/aggregate_main.py --results_dir results/main --eval_dir results/eval --out_prefix results/main"
Stage ablation  "python scripts/aggregate_ablation.py --results_dir results/main --out_csv results/ablation.csv"
# 3. robustness: ADS-B latency (jitter 0.4) and packet-loss sweeps, test split re-simulated per level
Stage robust    "python scripts/run_robustness.py --results_dir results/main --out_dir results"
# 4. scaling: fixed 5 km area, N = 100 ... 2000, log-log alpha fit (OOM rows recorded, not hidden)
Stage scaling   "python scripts/run_scaling.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/scaling.csv --out_fit results/scaling_fit.json"
# 5. attention vs AoI (last layer, in-degree normalised)
Stage attention "python scripts/analyze_attention_aoi.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/attention_vs_aoi.csv"
# 5b. S8e intent-conformance gate value per conflict cause
Stage gate      "python scripts/analyze_gate.py --checkpoint results/main/TR-GAT/seed42 --out_csv results/gate_by_cause.csv"
# 5c. S8f event-level (operational) metrics for every checkpoint: event CDR, lead time, false episodes / UAV-h
Stage events    "python scripts/eval_events.py --results_dir results/main --out_dir results --persistence 1 3"
# 5d. CPU-only inference latency of TR-GAT (edge box without GPU); graph build is CPU already
Stage cpulat    "python scripts/eval_only.py --results_dir results/main --methods TR-GAT --seeds 42 --device cpu --skip_graph_build --latency_epochs 200 --out_dir results/eval_cpu"
# 6. S16 artefacts
Stage figures   "python scripts/make_figures.py --results_dir results --out_dir paper/figs"
Stage tables    "python scripts/make_tables.py --results_dir results --out_dir paper/tables"
Stage macros    "python scripts/make_macros.py --results_dir results --out paper/numbers.tex"
Write-Host ("S15 finished in {0:N1} min" -f ((Get-Date) - $t0).TotalMinutes)
