#!/bin/bash

# Usage: ./run_pipeline.sh configs/your_config.yaml

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
LOG_FILE="${EXP_DIR}/${EXP_NAME}.log"
H5_FILE="${EXP_DIR}/${EXP_NAME}.h5"

# 2. Prepare Directory
echo "Initializing experiment: $EXP_NAME"
mkdir -p "$EXP_DIR"

# 3. Start Logging
echo "==========================================" > "$LOG_FILE"
echo "  Experiment: $EXP_NAME" >> "$LOG_FILE"
echo "  Date:       $(date)" >> "$LOG_FILE"
echo "  Config:     $CONFIG_FILE" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 1: PYTHON TRAINING
# ------------------------------------------------------------------------------
echo "Step 1/4: Running Python Training..." | tee -a "$LOG_FILE"

uv run run_training.py "$CONFIG_FILE" 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Python training failed!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "------------------------------------------" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 2: JULIA GEOMETRY ANALYSIS
# ------------------------------------------------------------------------------
echo "Step 2/4: Running Julia Geometry Analysis..." | tee -a "$LOG_FILE"

julia --threads $THREADS run_geobin.jl "$CONFIG_FILE" --overwrite 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Julia analysis failed!" | tee -a "$LOG_FILE"
    exit 1
fi
echo "------------------------------------------" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 3: METRIC PLOTS
# ------------------------------------------------------------------------------
echo "Step 3/4: Plotting Training Metrics..." | tee -a "$LOG_FILE"

uv run python visualization/plot_metrics.py "$H5_FILE" 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Metric plotting failed." | tee -a "$LOG_FILE"
else
    echo "✅ Metrics plotted." | tee -a "$LOG_FILE"
fi
echo "------------------------------------------" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 4: REGION VISUALIZATION
# ------------------------------------------------------------------------------
echo "Step 4/4: Visualizing Geometric Regions..." | tee -a "$LOG_FILE"

# Adjusted bound to 2.5 for a standard -2.5 to 2.5 view
uv run python visualization/plot_regions.py "$H5_FILE" --bound 2.5 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "❌ Region visualization failed." | tee -a "$LOG_FILE"
else
    echo "✅ Regions visualized." | tee -a "$LOG_FILE"
fi

echo "==========================================" >> "$LOG_FILE"
echo "  Pipeline Finished: $(date)" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

echo "Results available in: $EXP_DIR"