#!/usr/bin/env bash
# ============================================================
# Reproduce Table 7 — Scalability & Latency Analysis
# Paper: "SkyFlow: Temporal Relational Graph Attention for
#         Real-Time UAV Conflict Detection"
#
# Sweeps fleet sizes [100, 200, 300, 400, 500] and measures
# 95th-percentile latency breakdown:
#   - Graph construction (TKG Builder)
#   - TR-GAT forward pass
#   - Total end-to-end
#
# Legacy (pre-S8) entry point; the current pipeline is scripts/run_main.py + run.sh (see README).
# Result numbers are never typed here: read results/*.csv or paper/tables/*.tex.
#
# Estimated runtime: ~30 min on A100
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
CHECKPOINT=${CHECKPOINT:-outputs/best_model.pt}

echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Reproducing Table 7: Scalability & Latency Analysis    ║"
echo "║  SkyFlow                                                 ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Device:      $DEVICE"
echo "  Checkpoint:  $CHECKPOINT"
echo "  Fleet sizes: 100, 200, 300, 400, 500"
echo ""

if [ ! -f "$CHECKPOINT" ]; then
    echo "[INFO] No checkpoint found at $CHECKPOINT."
    echo "       Running with random weights for latency measurement."
    echo "       (To use trained weights, run reproduce_table3.sh first.)"
    echo ""
fi

echo "Starting scalability sweep..."
echo ""

$PYTHON scripts/eval_scalability.py \
    --config configs/default.yaml \
    --checkpoint "$CHECKPOINT" \
    --device "$DEVICE" \
    --n-epochs 1000

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  Table 7 reproduction complete."
echo "  Results saved to: outputs/scalability_results.json"
echo "════════════════════════════════════════════════════════════"
