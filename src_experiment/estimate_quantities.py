from __future__ import annotations
import numpy as np
import pandas as pd
import h5py
import sys
from pathlib import Path
from typing import Dict, List, Union

# Make sure geobin_py is available
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

try:
    from geobin_py.reconstruction import Tree
except ImportError:
    print("⚠️  Warning: Could not import 'geobin_py.reconstruction.Tree'.")

from .divergence_engine import DivergenceEngine, QUANTITIES_TO_ESTIMATE

class ExperimentEvaluator:
    """
    Evaluates a single experiment (.h5 file). 
    Reconstructs the trees, assigns points to regions, 
    and computes information-theoretic divergences.
    """

    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = Path(h5_path)
        self.estimates: Dict[str, List[pd.DataFrame]] = {
            q: [] for q in QUANTITIES_TO_ESTIMATE
        }
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
            
        self._evaluate()

    def _evaluate(self) -> None:
        print(f"Evaluating {self.h5_path.name}...")
        
        # 1. Load points and labels
        with h5py.File(self.h5_path, "r") as f:
            if "points" not in f:
                raise KeyError(f"No 'points' found in {self.h5_path.name}")
                
            points = f["points"][:]
            if len(points.shape) == 2 and points.shape[0] < points.shape[1]:
                points = points.T  # Fix Julia column-major shape
                
            labels = f["labels"][:] if "labels" in f else np.zeros(len(points))
            
            # Determine available epochs
            epochs = sorted([int(k.split("_")[1]) for k in f.get("epochs", {}).keys()])

        # 2. Iterate through epochs
        for ep in epochs:
            try:
                tree = Tree(str(self.h5_path), epoch=ep)
                if getattr(tree, 'root', None) is None:
                    continue # Tree not computed yet
                
                # Compute point counts
                df_counts = self._compute_region_counts(tree, points, labels)
                
                # Pass to Divergence Engine
                engine = DivergenceEngine(df_counts)
                epoch_results = engine.compute()

                # Store Results
                for key, df in epoch_results.items():
                    df.insert(0, "epoch", ep)
                    self.estimates[key].append(df)
                    
            except Exception as e:
                print(f"  ⚠️ Skipping epoch {ep}: {e}")

        # 3. Concatenate all epochs into unified DataFrames
        for key, frames in self.estimates.items():
            if frames:
                self.estimates[key] = (
                    pd.concat(frames, ignore_index=True)
                    .rename_axis(None, axis=1)
                )

    def _compute_region_counts(self, tree: Tree, points: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        """Pushes points through the tree and builds the count DataFrame."""
        all_layer_data = []
        unique_classes = np.unique(labels)
        
        for layer in range(1, tree.L + 1):
            assignments = tree.assign_points_to_regions(points, target_layer=layer)
            unique_regions = np.unique(assignments)
            
            for r_id in unique_regions:
                # Mask for points in this specific region
                # Note: We INTENTIONALLY include r_id == -1 to conserve total probability mass!
                in_region = (assignments == r_id)
                total_in_region = np.sum(in_region)
                
                row_data = {
                    "layer_idx": layer,
                    "region_idx": r_id,
                }
                
                # Count points by class
                for cls in unique_classes:
                    class_count = np.sum(in_region & (labels == cls))
                    row_data[str(int(cls))] = class_count
                    
                row_data["total"] = total_in_region
                all_layer_data.append(row_data)

        return pd.DataFrame(all_layer_data)

    def get_estimates(self) -> Dict[str, pd.DataFrame]:
        return self.estimates


class AverageEstimates:
    """
    Higher-level operator aggregating results over multiple experiment runs (.h5 files).
    """

    def __init__(self, h5_paths: List[Union[str, Path]]):
        self.h5_paths = [Path(p) for p in h5_paths]
        self.means = {q: [] for q in QUANTITIES_TO_ESTIMATE}
        self.stds = {q: [] for q in QUANTITIES_TO_ESTIMATE}
        self._find_means_and_stds()

    def _collect(self) -> Dict[str, List[pd.DataFrame]]:
        aggregated = {q: [] for q in QUANTITIES_TO_ESTIMATE}

        for path in self.h5_paths:
            evaluator = ExperimentEvaluator(h5_path=path)
            
            for key, df in evaluator.get_estimates().items():
                if isinstance(df, pd.DataFrame) and not df.empty:
                    aggregated[key].append(df)

        return aggregated
    
    def _concatenate(self) -> Dict[str, pd.DataFrame]:
        aggregated = self._collect()
        concatenated = {}

        for key, dfs in aggregated.items():
            if dfs:
                concatenated[key] = pd.concat(dfs, keys=np.arange(len(dfs)), names=["run", "row"]).reset_index(level="run")

        return concatenated
    
    def _find_means_and_stds(self) -> None:
        concatenated = self._concatenate()

        for key, df in concatenated.items():
            # Get columns dynamically, ignoring string columns
            numeric_cols = [col for col in df.columns if col not in ["run", "epoch"]]
            
            summary = (df
                       .groupby("epoch")[numeric_cols]
                       .agg(["mean", "std"])
                       )

            self.means[key] = summary.xs("mean", axis=1, level=1).reset_index()
            self.stds[key] = summary.xs("std", axis=1, level=1).reset_index()