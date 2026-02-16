#!/bin/bash

# Usage: ./run_computation.sh configs/your_config.yaml

CONFIG_FILE=$1
THREADS=$2

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <path_to_config.yaml>"
    exit 1
fi

# 1. Extract parameters from YAML
EXP_NAME=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['experiment_name'])")
OUTPUT_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output_dir'])")

# Define paths
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
LOG_FILE="${EXP_DIR}/${EXP_NAME}_comp.log"

# 2. Prepare Directory
echo "Initializing experiment: $EXP_NAME"
mkdir -p "$EXP_DIR"

# 3. Start Logging
echo "==========================================" > "$LOG_FILE"
echo "  Computation Run: $EXP_NAME" >> "$LOG_FILE"
echo "  Date:            $(date)" >> "$LOG_FILE"
echo "  Config:          $CONFIG_FILE" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 1: PYTHON TRAINING
# ------------------------------------------------------------------------------
echo "Step 1/2: Running Python Training..." | tee -a "$LOG_FILE"

uv run run_training.py "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Python training failed! Check $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi
echo "------------------------------------------" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 2: JULIA GEOMETRY ANALYSIS
# ------------------------------------------------------------------------------
echo "Step 2/2: Running Julia Geometry Analysis..." | tee -a "$LOG_FILE"

# Setting threads to auto for max performance during tree construction
julia --threads $THREADS run_geobin.jl "$CONFIG_FILE" --overwrite 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Julia analysis failed! Check $LOG_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "==========================================" >> "$LOG_FILE"
echo "  Computation Finished: $(date)" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

echo "HDF5 data generated and saved in: $EXP_DIR"