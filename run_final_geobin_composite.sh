#!/bin/bash

# Exit script immediately if any command fails
set -e

# Prevent the loop from running if no .h5 files are found (avoids literal passing of '*')
shopt -s nullglob

# Quoting the directory path is cleaner and safer than backslash-escaping spaces and brackets
for f in "outputs/composite_label_noise/n0.0_[25, 25, 25, 25, 25]"/*.h5; do 
    echo "Processing $f..."
    julia --threads 16 run_geobin.jl "$f" --overwrite --exact_regions
done

for f in "outputs/composite_label_noise/n0.0_[25, 25, 25, 25]"/*.h5; do 
    echo "Processing $f..."
    julia --threads 16 run_geobin.jl "$f" --overwrite --exact_regions
done

echo "All tasks completed successfully!"