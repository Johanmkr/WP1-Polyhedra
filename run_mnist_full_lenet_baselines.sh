#!/usr/bin/env bash
# Phase A.1 baseline sweep on the mnist_full_lenet HDF5s.
#
# Sweep targets the deepest hidden FC layer (layer = n_conv + n_fc_hidden = 4
# for every LeNet variant in this repo) at the last saved epoch, across the
# 60 trained HDF5s under outputs/mnist_full_lenet/ (4 archs * 3 noise * 5
# seeds). Outputs:
#   - per-HDF5: mi_baselines_seed_<seed>.csv next to each .h5 (resumable)
#   - aggregate: results/mi_baselines.csv (composite + wbc + mnist_full_lenet)
#   - figure:    figures/baseline_mi_with_plugin.png (3 panels once mnist
#                lands; auto-extended via DATASET_ORDER in the plot script)
#
# Env switches (set to 1 to enable):
#   MNIST_NO_PREFLIGHT  - skip the 1-cell smoke run.
#   MNIST_SKIP_MINE     - drop MINE-f from the sweep (saves ~most of the wall).
#   MNIST_SKIP_INFONCE  - drop InfoNCE.
#   MNIST_DRY_RUN       - --list jobs and exit (no work).
#   MNIST_NO_REPLOT     - skip the final figure regeneration step.
#
# Pass-through args (after `--`) go to scripts/run_mi_baselines.py for the
# main sweep step. Example:
#   ./run_mnist_full_lenet_baselines.sh -- --noise 0.0 0.4
#
# Reference: planning/phase_a_mnist_baselines.md.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

LOG_DIR="$REPO/logs"
mkdir -p "$LOG_DIR"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/mnist_full_lenet_baselines_${TIMESTAMP}.log"

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
section() {
    {
        echo
        echo "=== $* ==="
    } | tee -a "$LOG_FILE"
}

run() {
    echo "+ $*" | tee -a "$LOG_FILE"
    "$@" 2>&1 | tee -a "$LOG_FILE"
}

# ---------------------------------------------------------------------------
# Pre-flight: 1-cell smoke run with cheap baselines only.
# ---------------------------------------------------------------------------
if [[ "${MNIST_NO_PREFLIGHT:-0}" != "1" ]]; then
    section "Pre-flight: LeNet-XS / noise=0 / seed=101 (cheap baselines)"
    run uv run python scripts/run_mi_baselines.py \
        --datasets mnist_full_lenet \
        --noise 0.0 \
        --archs LeNet-XS \
        --seeds 101 \
        --skip-mine --skip-infonce \
        --force
fi

# ---------------------------------------------------------------------------
# Dry run: list jobs and exit.
# ---------------------------------------------------------------------------
if [[ "${MNIST_DRY_RUN:-0}" == "1" ]]; then
    section "Dry run: listing jobs"
    run uv run python scripts/run_mi_baselines.py \
        --datasets mnist_full_lenet \
        --list
    exit 0
fi

# ---------------------------------------------------------------------------
# Main sweep.
# ---------------------------------------------------------------------------
SWEEP_FLAGS=()
if [[ "${MNIST_SKIP_MINE:-0}" == "1" ]]; then
    SWEEP_FLAGS+=(--skip-mine)
fi
if [[ "${MNIST_SKIP_INFONCE:-0}" == "1" ]]; then
    SWEEP_FLAGS+=(--skip-infonce)
fi

# Forward any user-supplied flags after `--`.
PASSTHROUGH=()
if [[ "${1:-}" == "--" ]]; then
    shift
    PASSTHROUGH=("$@")
fi

section "Main sweep: 60 jobs (4 archs * 3 noise * 5 seeds)"
run uv run python scripts/run_mi_baselines.py \
    --datasets mnist_full_lenet \
    "${SWEEP_FLAGS[@]}" \
    "${PASSTHROUGH[@]}"

# ---------------------------------------------------------------------------
# Aggregate (re-aggregates composite + wbc as a side effect, picking up the
# new bits_ours_plugin / bits_ours_func_plugin columns).
# ---------------------------------------------------------------------------
section "Aggregate: results/mi_baselines.csv"
run uv run python scripts/run_mi_baselines.py --aggregate

# ---------------------------------------------------------------------------
# Replot. The plot script's DATASET_ORDER currently lists composite + wbc;
# add 'mnist_full_lenet' there to enable a 3-panel figure (one-line edit
# documented at the top of the script).
# ---------------------------------------------------------------------------
if [[ "${MNIST_NO_REPLOT:-0}" != "1" ]]; then
    section "Replot: figures/baseline_mi_with_plugin.png"
    run uv run python scripts/plot_routing_vs_baselines.py
fi

section "Done"
echo "Log: $LOG_FILE"
