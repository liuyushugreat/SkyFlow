#!/usr/bin/env bash
# ============================================================
# Reproduce Table 3 — Overall Detection Performance & Latency
# Paper: "SkyFlow: Temporal Relational Graph Attention for
#         Real-Time UAV Conflict Detection"
#
# This script trains TR-GAT and all 6 baselines across 5 seeds,
# then prints the comparison table matching Table 3 in the paper.
#
# Legacy (pre-S8) entry point; the current pipeline is scripts/run_main.py + run.sh (see README).
# Result numbers are never typed here: read results/*.csv or paper/tables/*.tex.
#
# Estimated runtime: ~14 hours on A100, ~5 min with --quick
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

if command -v python3 &>/dev/null; then
    PYTHON=${PYTHON:-python3}
elif command -v python &>/dev/null; then
    PYTHON=${PYTHON:-python}
else
    echo "[ERROR] Neither python3 nor python found in PATH."
    exit 1
fi
DEVICE=${DEVICE:-auto}
QUICK=${QUICK:-false}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Reproducing Table 3: Overall Detection Performance     ║"
echo "║  SkyFlow                                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Device:  $DEVICE"
echo "  Config:  configs/default.yaml"
echo "  Seeds:   42, 123, 456, 789, 1024"
echo ""

ARGS="--config configs/default.yaml --device $DEVICE"
if [ "$QUICK" = "true" ]; then
    echo "  [!] Quick mode enabled (reduced UAVs & epochs)"
    ARGS="$ARGS --quick"
fi

echo "Starting full pipeline (data → training → baselines → figures)..."
echo ""

$PYTHON scripts/reproduce_paper.py $ARGS

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Table 3 reproduction complete."
echo "  Results saved to: outputs/all_results.json"
echo "  Figures saved to: outputs/charts/"
echo "════════════════════════════════════════════════════════════"
