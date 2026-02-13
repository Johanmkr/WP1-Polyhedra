#!/bin/bash

# Usage: ./run_visualization.sh configs/your_config.yaml [bound_value]
# Example: ./run_visualization.sh configs/moons.yaml 5.0

CONFIG_FILE=$1
BOUND=${2:-2.5}  # Uses $2 if provided, otherwise defaults to 2.5

if [ -z "$CONFIG_FILE" ]; then
    echo "Usage: $0 <path_to_config.yaml> [bound]"
    exit 1
fi

# 1. Extract parameters from YAML
EXP_NAME=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['experiment_name'])")
OUTPUT_DIR=$(uv run python -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['output_dir'])")

# Define paths
EXP_DIR="${OUTPUT_DIR}/${EXP_NAME}"
LOG_FILE="${EXP_DIR}/${EXP_NAME}_viz.log"
H5_FILE="${EXP_DIR}/${EXP_NAME}.h5"

# 2. Check if HDF5 file exists
if [ ! -f "$H5_FILE" ]; then
    echo "❌ Error: HDF5 file not found at $H5_FILE"
    exit 1
fi

# 3. Start Logging
echo "==========================================" > "$LOG_FILE"
echo "  Visualization Run: $EXP_NAME" >> "$LOG_FILE"
echo "  Bound Setting:     $BOUND" >> "$LOG_FILE"
echo "  Date:              $(date)" >> "$LOG_FILE"
echo "==========================================" >> "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 1: METRIC PLOTS
# ------------------------------------------------------------------------------
echo "Step 1/2: Plotting Training Metrics..." | tee -a "$LOG_FILE"
uv run python visualization/plot_metrics.py "$H5_FILE" 2>&1 | tee -a "$LOG_FILE"

# ------------------------------------------------------------------------------
# STEP 2: REGION VISUALIZATION
# ------------------------------------------------------------------------------
echo "Step 2/2: Visualizing Geometric Regions (Bound: $BOUND)..." | tee -a "$LOG_FILE"

# We pass the $BOUND variable directly to the python script
uv run python visualization/plot_regions.py "$H5_FILE" --bound "$BOUND" 2>&1 | tee -a "$LOG_FILE"

if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo "✅ Visualization successful. Bound used: $BOUND" | tee -a "$LOG_FILE"
else
    echo "❌ Region visualization failed." | tee -a "$LOG_FILE"
fi

echo "Results available in: $EXP_DIR"