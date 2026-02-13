import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
from pathlib import Path

# Add project root to path just in case we need project modules later
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

def plot_metrics(h5_file):
    file_path = Path(h5_file).resolve()
    if not file_path.exists():
        print(f"❌ Error: File not found at {file_path}")
        sys.exit(1)

    print(f"Loading metrics from {file_path.name}...")
    
    experiment_name = "Training Results"
    
    with h5py.File(file_path, 'r') as f:
        # 1. Load Config (Metadata)
        print("\n--- Experiment Configuration ---")
        if 'metadata' in f:
            # Try to get experiment name for the plot title
            if 'experiment_name' in f['metadata'].attrs:
                val = f['metadata'].attrs['experiment_name']
                experiment_name = val.decode('utf-8') if isinstance(val, bytes) else val

            for k, v in f['metadata'].attrs.items():
                val = v
                if isinstance(v, bytes):
                    val = v.decode('utf-8')
                print(f"{k}: {val}")
        
        # 2. Load Metrics
        if 'training_results' not in f:
            print("\n❌ No training results found in file.")
            return

        results = f['training_results']
        keys = list(results.keys())
        
        data = {}
        for k in keys:
            data[k] = results[k][:]
            
        # Assume all metrics have the same length (epochs)
        epochs = np.arange(1, len(data[keys[0]]) + 1)
        
        # 3. Print Final Summary
        print("\n--- Final Epoch Results ---")
        for k, v in data.items():
            print(f"{k}: {v[-1]:.4f}")

    # ==========================================================================
    # 4. PLOTTING (Twin Axis Style)
    # ==========================================================================
    
    # Setup the figure
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()  # Instantiate a second axes that shares the same x-axis

    # Define Style Logic
    # train -> solid, test -> dashed
    # loss -> blue, accuracy -> red
    
    def get_style(metric_name):
        style = '-' # default solid
        if 'test' in metric_name or 'val' in metric_name:
            style = '--'
        return style

    has_loss = False
    has_acc = False

    # Plot specific known keys to match desired order/colors
    # 1. LOSS (Primary Axis, Blue)
    if 'train_loss' in data:
        ax1.plot(epochs, data['train_loss'], color='blue', linestyle='-', label='Train Loss')
        has_loss = True
    if 'test_loss' in data:
        ax1.plot(epochs, data['test_loss'], color='blue', linestyle='--', label='Test Loss')
        has_loss = True

    # 2. ACCURACY (Secondary Axis, Red)
    if 'train_accuracy' in data:
        ax2.plot(epochs, data['train_accuracy'], color='red', linestyle='-', label='Train Acc')
        has_acc = True
    if 'test_accuracy' in data:
        ax2.plot(epochs, data['test_accuracy'], color='red', linestyle='--', label='Test Acc')
        has_acc = True

    # Fallback: Loop through other keys if they weren't caught above
    # (e.g. 'eval_train_loss', 'custom_metric')
    processed_keys = {'train_loss', 'test_loss', 'train_accuracy', 'test_accuracy'}
    for k in data.keys():
        if k in processed_keys:
            continue
        
        # Determine axis and color dynamically
        if 'loss' in k:
            ax1.plot(epochs, data[k], color='blue', alpha=0.5, linestyle=get_style(k), label=k)
            has_loss = True
        elif 'acc' in k:
            ax2.plot(epochs, data[k], color='red', alpha=0.5, linestyle=get_style(k), label=k)
            has_acc = True
        else:
            # Unknown metric type, put on primary axis with gray
            ax1.plot(epochs, data[k], color='gray', linestyle=get_style(k), label=k)

    # Labeling
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color='blue')
    ax2.set_ylabel("Accuracy", color='red')
    
    ax1.tick_params(axis='y', labelcolor='blue')
    ax2.tick_params(axis='y', labelcolor='red')
    
    ax1.set_title(f"Training Results: {experiment_name}")
    ax1.grid(True, alpha=0.3)

    # Combined Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    
    # Place legend in "center right" or "best"
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    # Save logic
    output_plot = file_path.with_suffix('.png')
    
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150)
    print(f"\n✅ Plot saved to: {output_plot}")
    # plt.show() # Uncomment if running locally with a display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training metrics from HDF5.")
    parser.add_argument("file", type=str, help="Path to experiment .h5 file")
    
    args = parser.parse_args()
    plot_metrics(args.file)