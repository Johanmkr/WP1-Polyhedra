import argparse
import sys
from pathlib import Path

# Ensure src_experiment is importable
sys.path.append(str(Path(__file__).resolve().parent))

from src_experiment.run_experiment import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run experiment pipeline.")
    parser.add_argument("config", type=str, help="Path to the YAML configuration file.")
    
    args = parser.parse_args()
    
    # Execute the experiment using the provided config path
    run(args.config)