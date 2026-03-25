import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
# Assuming this is run from the project root or src_experiment module




project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from src_experiment.dataset import get_new_data
    from src_experiment.paths import neurips_figpath
    from src_experiment.utils import savefig
except ImportError:
    print("❌ Error: Could not import 'geobin_py.reconstruction'.")
    sys.exit(1)
    
def plot_composite_dataset():
    print("Generating Composite Dataset...")
    
    # We load the dataset using the main registry function to ensure 
    # it goes through the exact same process_and_split (MinMax scaling to [-1, 1])
    # Setting batch_size to a large number to grab all points at once.
    train_loader, test_loader = get_new_data(
        dataset_name="composite", 
        split_seed=42, 
        batch_size=10000 
    )

    # Extract the training tensors
    X_train, y_train = next(iter(train_loader))
    X = X_train.numpy()
    y = y_train.numpy()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Scatter plot with a distinct colormap for 7 classes
    scatter = ax.scatter(
        X[:, 0], X[:, 1], 
        c=y, 
        cmap='tab10',  # tab10 is good for distinct categorical classes
        alpha=0.7, 
        s=15,
        edgecolors='none'
    )

    # Formatting
    ax.set_title("2D Composite Dataset (Scaled)", fontsize=16, pad=15)
    ax.set_xlabel(r"$x_1$", fontsize=14)
    ax.set_ylabel(r"$x_2$", fontsize=14)
    
    # Set limits slightly outside the [-1, 1] bounds for visual padding
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', 'box')

    # Add a legend
    legend = ax.legend(
        *scatter.legend_elements(), 
        title="Classes",
        loc="upper right",
        bbox_to_anchor=(1.15, 1)
    )
    ax.add_artist(legend)

    # Save using your custom savefig function from utils.py
    # This will generate both .png and .pdf automatically
    
    savefig(fig, neurips_figpath / "composite_dataset")
    plt.show()

if __name__ == "__main__":
    plot_composite_dataset()