#!/usr/bin/env bash
# Phase A baselines (`planning/phase_a_baselines.md`) — end-to-end runner.
#
# Steps
#   A.1 — MI baselines
#     1. Validate MI-baseline implementations (synthetic ground-truth gate).
#     2. Run the full MI-baseline sweep across every label-noise HDF5
#        (composite + wbc). Per-HDF5 CSVs land next to each .h5; the sweep is
#        resumable.
#     3. Aggregate the per-HDF5 CSVs into results/mi_baselines.csv.
#
#   A.2 — Generalization-gap predictors
#     4. Validate gen-gap-predictor implementations (closed-form gate).
#     5. Run the full gen-gap-predictor sweep across every label-noise HDF5.
#     6. Aggregate the per-HDF5 CSVs into results/gen_gap_predictors.csv.
#
# Plotting and §5 writeup are post-aggregation steps — added below as
# they come online.
#
# Usage:
#   ./run_phase_a_baselines.sh                    # full A.1 + A.2 sweep
#   ./run_phase_a_baselines.sh --datasets wbc     # one dataset only
#   PHASE_A_NO_VALIDATE=1 ./run_phase_a_baselines.sh   # skip both gates
#   PHASE_A_SKIP_MI=1   ./run_phase_a_baselines.sh     # skip A.1
#   PHASE_A_SKIP_GG=1   ./run_phase_a_baselines.sh     # skip A.2
#
# A.1-only fast tier (binning+kmeans+KSG; no MINE/InfoNCE):
#   PHASE_A_SKIP_GG=1 ./run_phase_a_baselines.sh --skip-mine --skip-infonce
#
# Logs to logs/phase_a_baselines_YYYYMMDD_HHMMSS.log.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$REPO/logs"
LOG_FILE="$LOG_DIR/phase_a_baselines_${TS}.log"
mkdir -p "$LOG_DIR" "$REPO/results"

export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

PYTHON="uv run python"
RUN_ARGS=("$@")

banner() {
    echo "=========================================="
    echo "  $1"
    echo "  $(date)"
    echo "=========================================="
}

# -- Header ------------------------------------------------------------------
{
    banner "Phase A baselines — runner start"
    echo "  repo:    $REPO"
    echo "  log:     $LOG_FILE"
    echo "  args:    ${RUN_ARGS[*]:-<none>}"
    echo "  threads: OPENBLAS=$OPENBLAS_NUM_THREADS MKL=$MKL_NUM_THREADS OMP=$OMP_NUM_THREADS"
    echo "  flags:   PHASE_A_NO_VALIDATE=${PHASE_A_NO_VALIDATE:-0}"
    echo "           PHASE_A_SKIP_MI=${PHASE_A_SKIP_MI:-0}"
    echo "           PHASE_A_SKIP_GG=${PHASE_A_SKIP_GG:-0}"
} | tee -a "$LOG_FILE"

# ============================================================================
# A.1 — MI baselines
# ============================================================================
if [[ "${PHASE_A_SKIP_MI:-0}" != "1" ]]; then
    if [[ "${PHASE_A_NO_VALIDATE:-0}" != "1" ]]; then
        banner "Step 1/6 — Validate MI baselines (synthetic gate)" | tee -a "$LOG_FILE"
        if ! $PYTHON scripts/validate_mi_baselines.py 2>&1 | tee -a "$LOG_FILE"; then
            echo "MI-baseline validation FAILED. Aborting." | tee -a "$LOG_FILE"
            exit 1
        fi
    else
        echo "PHASE_A_NO_VALIDATE=1 — skipping MI-baseline gate." | tee -a "$LOG_FILE"
    fi

    banner "Step 2/6 — Run MI-baseline sweep" | tee -a "$LOG_FILE"
    $PYTHON scripts/run_mi_baselines.py "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

    banner "Step 3/6 — Aggregate MI-baseline CSVs" | tee -a "$LOG_FILE"
    $PYTHON scripts/run_mi_baselines.py --aggregate \
        --output "$REPO/results/mi_baselines.csv" 2>&1 | tee -a "$LOG_FILE"
else
    echo "PHASE_A_SKIP_MI=1 — skipping all of A.1." | tee -a "$LOG_FILE"
fi

# ============================================================================
# A.2 — Generalization-gap predictors
# ============================================================================
if [[ "${PHASE_A_SKIP_GG:-0}" != "1" ]]; then
    if [[ "${PHASE_A_NO_VALIDATE:-0}" != "1" ]]; then
        banner "Step 4/6 — Validate gen-gap predictors (closed-form gate)" | tee -a "$LOG_FILE"
        if ! $PYTHON scripts/validate_gen_gap_predictors.py 2>&1 | tee -a "$LOG_FILE"; then
            echo "Gen-gap-predictor validation FAILED. Aborting." | tee -a "$LOG_FILE"
            exit 1
        fi
    else
        echo "PHASE_A_NO_VALIDATE=1 — skipping gen-gap gate." | tee -a "$LOG_FILE"
    fi

    banner "Step 5/6 — Run gen-gap-predictor sweep" | tee -a "$LOG_FILE"
    $PYTHON scripts/run_gen_gap_predictors.py "${RUN_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"

    banner "Step 6/6 — Aggregate gen-gap-predictor CSVs" | tee -a "$LOG_FILE"
    $PYTHON scripts/run_gen_gap_predictors.py --aggregate \
        --output "$REPO/results/gen_gap_predictors.csv" 2>&1 | tee -a "$LOG_FILE"
else
    echo "PHASE_A_SKIP_GG=1 — skipping all of A.2." | tee -a "$LOG_FILE"
fi

# -- Footer ------------------------------------------------------------------
{
    banner "Phase A baselines — done"
    echo "  log:     $LOG_FILE"
    echo "  outputs:"
    echo "    $REPO/results/mi_baselines.csv"
    echo "    $REPO/results/gen_gap_predictors.csv"
} | tee -a "$LOG_FILE"
