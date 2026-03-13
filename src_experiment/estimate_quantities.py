from __future__ import annotations
import numpy as np
import pandas as pd
import h5py
import sys
from pathlib import Path
from typing import Dict, Union, Optional, List

# Make sure geobin_py is available
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from geobin_py.reconstruction import Tree
except ImportError:
    print("⚠️  Warning: Could not import 'geobin_py.reconstruction.Tree'.")


class ExperimentEvaluator:
    """
    Evaluates an entire experiment. 
    Loads data, discovers all epochs, reconstructs the trees, routes points, 
    estimates empirical Mutual Information I(Y;W) and I(X;W) in bits,
    and tracks the total number of regions.
    """

    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)
        
        # Data attributes
        self.points: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.N_total: int = 0
        self.unique_classes: Optional[np.ndarray] = None
        self.epochs: List[int] = []
        
        # Results
        self.results_df: Optional[pd.DataFrame] = None

    def load_data(self) -> None:
        """Loads points, labels, and discovers available epochs from the HDF5 file."""
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
            
        with h5py.File(self.h5_path, "r") as f:
            if "points" not in f:
                raise KeyError(f"No 'points' found in {self.h5_path.name}")
                
            points = f["points"][:]
            
            # Fix Julia column-major shape if necessary
            if len(points.shape) == 2 and points.shape[0] < points.shape[1]:
                points = points.T
                
            labels = f["labels"][:] if "labels" in f else np.zeros(len(points))
            
            # Discover available epochs
            if "epochs" in f:
                self.epochs = sorted([int(k.split("_")[1]) for k in f["epochs"].keys()])
            else:
                print("⚠️ No 'epochs' group found in HDF5 file.")
            
        self.points = points
        self.labels = labels
        self.N_total = len(points)
        self.unique_classes = np.unique(labels)
        
        print(f"Loaded {self.N_total} points. Found {len(self.epochs)} epochs.")

    def _build_tree(self, epoch: int) -> Optional['Tree']:
        """Instantiates the geobin_py Tree for a specific epoch."""
        tree = Tree(str(self.h5_path), epoch=epoch)
        if getattr(tree, 'root', None) is None:
            return None
        return tree

    def _compute_region_counts(self, tree: 'Tree') -> pd.DataFrame:
        """
        Routes points through the tree and aggregates counts per region per class.
        """
        counts_dict = tree.perform_number_count(self.points, y=self.labels)
        all_layer_data = []

        for layer_idx, regions in counts_dict.items():
            if layer_idx == 0:
                continue 
                
            for r_id, class_counts in regions.items():
                total_in_region = sum(class_counts.values())
                
                # We only care about populated regions
                if total_in_region > 0:
                    row_data = {
                        "layer_idx": layer_idx,
                        "region_idx": r_id,
                        "total": total_in_region
                    }
                    # Add individual class counts dynamically
                    for cls in self.unique_classes:
                        row_data[str(int(cls))] = class_counts.get(cls, 0)
                        
                    all_layer_data.append(row_data)

        return pd.DataFrame(all_layer_data)

    def _estimate_mi_y_w(self, df_counts: pd.DataFrame) -> Dict[int, float]:
        """
        Estimates I(Y;W): The predictive capacity of the regions (in Bits).
        """
        class_cols = [str(int(c)) for c in self.unique_classes]
        mi_per_layer = {}
        
        for layer_idx, group in df_counts.groupby("layer_idx"):
            P_yw = group[class_cols].to_numpy() / self.N_total
            P_w = group["total"].to_numpy() / self.N_total
            P_y = P_yw.sum(axis=0)
            
            denominator = P_w[:, None] * P_y[None, :]
            mask = P_yw > 0
            
            kl_terms = np.zeros_like(P_yw)
            kl_terms[mask] = P_yw[mask] * np.log2(P_yw[mask] / denominator[mask])
            
            mi_per_layer[layer_idx] = kl_terms.sum()
            
        return mi_per_layer

    def _estimate_mi_x_w(self, df_counts: pd.DataFrame) -> Dict[int, float]:
        """
        Estimates I(X;W): The memorization capacity of the regions (in Bits).
        Since H(W|X) = 0 for deterministic mappings, I(X;W) = H(W).
        """
        mi_per_layer = {}
        
        for layer_idx, group in df_counts.groupby("layer_idx"):
            P_w = group["total"].to_numpy() / self.N_total
            
            mask = P_w > 0
            entropy_terms = np.zeros_like(P_w)
            entropy_terms[mask] = -P_w[mask] * np.log2(P_w[mask])
            
            mi_per_layer[layer_idx] = entropy_terms.sum()
            
        return mi_per_layer

    def evaluate_all(self) -> pd.DataFrame:
        """
        Runs the entire pipeline across all epochs discovered in the HDF5 file.
        Returns a DataFrame containing MI estimates and region counts per epoch and per layer.
        """
        if self.points is None:
            self.load_data()
            
        all_results = []

        for ep in self.epochs:
            print(f"Processing epoch {ep}...")
            try:
                # 1. Build the tree
                tree = self._build_tree(ep)
                if tree is None:
                    print(f"  ⚠️ Skipping epoch {ep}: Tree could not be built.")
                    continue
                
                # 2. Route the points once
                df_counts = self._compute_region_counts(tree)
                if df_counts.empty:
                    print(f"  ⚠️ Skipping epoch {ep}: No regions populated.")
                    continue
                
                # 3. Compute both MIs
                mi_yw = self._estimate_mi_y_w(df_counts)
                mi_xw = self._estimate_mi_x_w(df_counts)
                
                # 4. Store layer-by-layer results
                for layer_idx in mi_yw.keys():
                    
                    # --- NEW: Get region counts ---
                    # Total instantiated regions at this layer
                    total_regions = len(tree.get_regions_at_layer(layer_idx))
                    
                    # Number of regions that actually contain data points
                    layer_data = df_counts[df_counts["layer_idx"] == layer_idx]
                    populated_regions = len(layer_data)
                    
                    all_results.append({
                        "epoch": ep,
                        "layer_idx": layer_idx,
                        "I(Y;W)": mi_yw.get(layer_idx, 0.0),
                        "I(X;W)": mi_xw.get(layer_idx, 0.0),
                        "total_regions": total_regions,
                        "populated_regions": populated_regions
                    })
                    
            except Exception as e:
                print(f"  ⚠️ Error evaluating epoch {ep}: {e}")

        # Consolidate into a clean Pandas DataFrame
        self.results_df = pd.DataFrame(all_results)
        print("Evaluation complete.")
        return self.results_df


# ==========================================
# Example Usage:
# ==========================================
if __name__ == "__main__":
    h5_path = "path/to/your/experiment.h5"
    
    # 1. Initialize with just the file path
    evaluator = ExperimentEvaluator(h5_path)
    
    # 2. Run the evaluation across all epochs
    df_results = evaluator.evaluate_all()
    
    # 3. Output the results dataframe
    print("\nFinal Results DataFrame:")
    print(df_results.head(15))
    
    # Bonus: You can now easily plot this using seaborn or matplotlib!
    # import seaborn as sns
    # import matplotlib.pyplot as plt
    # sns.lineplot(data=df_results, x="epoch", y="I(X;W)", hue="layer_idx")
    # plt.show()