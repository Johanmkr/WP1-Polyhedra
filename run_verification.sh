#!/bin/bash

# Usage: ./run_verification.sh <config.yaml> [bound] [threads]

CONFIG_FILE=$1
BOUND=${2:-10.0}
THREADS=${3:-16}

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <path_to_config.yaml> [bound] [threads]"
    exit 1
fi

# 1. Extract parameters from YAML (using uv/python like your other scripts)
EXP_NAME=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['experiment_name'])")
OUTPUT_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output_dir'])")

# Define paths
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
LOG_FILE="${EXP_DIR}/${EXP_NAME}_verif.log"
H5_FILE="${EXP_DIR}/${EXP_NAME}.h5"

# 2. Check if HDF5 file exists
if [ ! -f "$H5_FILE" ]; then
    echo "❌ Error: HDF5 file not found at $H5_FILE"
    exit 1
fi

# 3. Start Logging
echo "==========================================" > "$LOG_FILE"
echo "  Verification Run: $EXP_NAME" >> "$LOG_FILE"
echo "  Date:             $(date)" >> "$LOG_FILE"
echo "  Config:           $CONFIG_FILE" >> "$LOG_FILE"
echo "  Bound:            $BOUND" >> "$LOG_FILE"
echo "  Threads:          $THREADS" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# 4. Environment Setup
export JULIA_NUM_THREADS=$THREADS
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Step 1/1: Running Julia Verification Suite..." | tee -a "$LOG_FILE"

# 5. Run Verification with Logging
# We pipe both stdout (1) and stderr (2) to tee, which writes to screen + file
julia --project=. --threads $THREADS verify_h5.jl "$CONFIG_FILE" --bound "$BOUND" 2>&1 | tee -a "$LOG_FILE"

# 6. Check Status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "==========================================" >> "$LOG_FILE"
    echo "✅ Verification run complete." | tee -a "$LOG_FILE"
else
    echo "==========================================" >> "$LOG_FILE"
    echo "❌ Verification failed. Check log for details." | tee -a "$LOG_FILE"
    exit 1
fi

echo "Results available in: $EXP_DIR"