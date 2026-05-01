#!/usr/bin/env bash
# Phase C-M2 — drive the 60-net mnist_full LeNet sweep.
#
# Usage:
#     ./run_mnist_full_lenet_sweep.sh           # all 60
#     ./run_mnist_full_lenet_sweep.sh --resume  # skip configs whose HDF5 exists
#
# Logs to logs/mnist_full_lenet_<timestamp>.log. Resumable: any
# (arch, seed) whose HDF5 already exists is skipped (matches the
# behaviour of run_training.py without --overwrite).

set -u
cd "$(dirname "$0")"

TS=$(date +"%Y%m%d_%H%M%S")
LOG="logs/mnist_full_lenet_${TS}.log"
mkdir -p logs

resume=0
for arg in "$@"; do
  case "$arg" in
    --resume) resume=1 ;;
    -h|--help)
      echo "usage: $0 [--resume]" && exit 0 ;;
  esac
done

cfgs=$(uv run python scripts/generate_mnist_full_lenet_configs.py --list)
total=$(echo "$cfgs" | wc -l)
echo "[mnist_full_lenet sweep] total=${total} resume=${resume}" | tee -a "$LOG"

i=0
for cfg in $cfgs; do
  i=$((i + 1))
  exp_name=$(awk '/^experiment_name:/ {print $2}' "$cfg")
  out_dir=$(awk '/^output_dir:/ {print $2}' "$cfg")
  seed=$(awk '/^model_seed:/ {print $2}' "$cfg")
  h5="${out_dir}/${exp_name}/seed_${seed}.h5"

  if [[ $resume -eq 1 && -f "$h5" ]]; then
    echo "[$i/$total] skip (exists) $h5" | tee -a "$LOG"
    continue
  fi

  echo "[$i/$total] $cfg -> $h5" | tee -a "$LOG"
  uv run python run_training.py "$cfg" --overwrite >> "$LOG" 2>&1 \
    && echo "  ok" | tee -a "$LOG" \
    || echo "  FAIL ($?)" | tee -a "$LOG"
done

echo "[mnist_full_lenet sweep] done at $(date +"%Y-%m-%d %H:%M:%S")" | tee -a "$LOG"
