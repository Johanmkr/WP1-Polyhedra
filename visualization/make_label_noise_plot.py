import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Add project root to sys.path to ensure we can import local modules
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from geobin_py.reconstruction import Tree, Region
    from src_experiment.paths import outputs
    from src_experiment.estimate_quantities import ExperimentEvaluator
    from src_experiment.paths import neurips_figpath
    from src_experiment.utils import savefig
except ImportError as e:
    print(f"❌ Error: Could not import required local modules. Details: {e}")
    sys.exit(1)


def get_path(noise: float, architecture: str, seed: int, data) -> Path:
    """Generates the file path for a specific experiment configuration."""
    if data == "wbc":
        return outputs / f"wbc_label_noise/n{noise}_{architecture}/seed_{int(seed)}.h5"
    elif data == "comp":
        return outputs / f"composite_label_noise/n{noise}_{architecture}/seed_{int(seed)}.h5"


def get_mean_mi(data, noise: float, architecture: str, seeds: list[int] = None) -> pd.DataFrame:
    """
    Loads mutual information data across multiple seeds and calculates the mean.
    """
    if seeds is None:
        seeds = [101, 102, 103, 104, 105]
        
    dfs = []
    for seed in seeds:
        path = get_path(noise, architecture, seed, data)
        # Check if file exists before trying to load it to prevent crashes
        if not path.exists():
            print(f"Warning: Missing data file {path}")
            continue
            
        ev = ExperimentEvaluator(path)
        df_results = ev.evaluate_all()
        df_results["Y(I;W)"] = df_results["I(Y;W)"]  - df_results["MMcorr"]  # Apply Miller-Madow correction
        print(df_results["MMcorr"].head())
        dfs.append(df_results)
        
    if not dfs:
        return pd.DataFrame()

    # Concatenate all seed dataframes
    df_all = pd.concat(dfs, ignore_index=True)
    
    # Use pandas groupby to calculate the mean across all seeds in one shot
    df_mean = (
        df_all.groupby(["epoch", "layer_idx"])[["I(Y;W)", "I(X;W)"]]
        .mean()
        .reset_index()
    )
    
    return df_mean


def main(data="wbc"):
    noise_levels = [0.0,]# 0.2, 0.4]
    
    # Group the architectures by the number of hidden layers
    architecture_groups = {
        # "3_layers": ["[5, 5, 5]", "[7, 7, 7]", "[9, 9, 9]", "[25, 25, 25]"],
        # "4_layers": ["[5, 5, 5, 5]", "[7, 7, 7, 7]", "[9, 9, 9, 9]", "[25, 25, 25, 25]"],
        "5_layers": ["[5, 5, 5, 5, 5]", "[7, 7, 7, 7, 7]"]# "[9, 9, 9, 9, 9]", "[25, 25, 25, 25, 25]"]
    }

    for group_name, architectures in architecture_groups.items():
        print(f"\n--- Processing {group_name} ---")
        all_data = []

        print("Aggregating data...")
        for arch in architectures:
            for noise in noise_levels:
                df_mean = get_mean_mi(data, noise, arch)
                
                if not df_mean.empty:
                    # Inject metadata columns for Seaborn
                    df_mean['architecture'] = arch
                    df_mean['label_noise'] = noise 
                    all_data.append(df_mean)

        if not all_data:
            print(f"No data found for {group_name}, skipping plot.")
            continue

        # Combine everything into one master DataFrame for this specific plot
        combined_df = pd.concat(all_data, ignore_index=True)

        print("Generating plot...")
        # Generate the Seaborn plot
        # Identify the last layer for this specific depth group and filter it out
        max_layer = combined_df['layer_idx'].max()
        plot_df = combined_df[combined_df['layer_idx'] < max_layer]
        
    

        print("Generating plot...")
        # Generate the Seaborn plot using the filtered dataframe
        g = sns.relplot(
            data=plot_df, 
            x="epoch",
            y="I(Y;W)",
            hue="layer_idx",
            col="label_noise",
            row="architecture",
            kind="line",
            palette="Set1",
            height=3,
            aspect=1.5
        )

        # Update the y-axis label to the requested LaTeX string
        g.set_ylabels(r"I(Y;$\Omega$)")

        # Clean up the titles
        g.set_titles(row_template="{row_name} Architecture", col_template="Noise: {col_name}")

        # Clean up the titles
        g.set_titles(row_template="{row_name} Architecture", col_template="Noise: {col_name}")
        g.fig.subplots_adjust(top=0.9)
        
        # Give each plot a distinct title based on its depth
        # depth = group_name.split('_')[0]
        # g.fig.suptitle(f"Information Plane Dynamics - {depth} Hidden Layers")
        
        # Save each plot with a distinct filename
        filename = f"{data}_label_noise_exp_{group_name}.pdf"
        # plt.savefig(neurips_figpath / filename, dpi=300)
        # savefig(g.fig, neurips_figpath / filename)
        
        plt.show()
        
        # Close the figure to free memory before the next loop iteration
        plt.close(g.fig)


if __name__ == "__main__":
    main("comp")
    # main("wbc")
