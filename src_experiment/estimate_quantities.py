from __future__ import annotations
import numpy as np
import pandas as pd
import h5py
import sys
from pathlib import Path
from typing import Dict, List, Union, Optional

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
    Evaluates a single experiment (.h5 file). 
    Reconstructs the trees, assigns points to regions, 
    and directly computes Mutual Information (Kullback-Leibler divergence).
    """

    def __init__(self, h5_path: Union[str, Path], volume_key: Optional[str] = None):
        """
        Args:
            h5_path: Path to the .h5 experiment file.
            volume_key: If provided (e.g., "volume_ex" or "volume_es"), the MI calculation 
                        will reweight the region probabilities P(W) by their geometric volume 
                        instead of their empirical point counts.
        """
        self.h5_path = Path(h5_path)
        self.volume_key = volume_key
        self.estimates: Dict[str, List[pd.DataFrame]] = {
            "Mutual Information": []
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

        unique_classes = np.unique(labels)
        N_total = len(points)

        # 2. Iterate through epochs
        for ep in epochs:
            try:
                tree = Tree(str(self.h5_path), epoch=ep)
                if getattr(tree, 'root', None) is None:
                    continue # Tree not computed yet

                # Fetch volumes from HDF5 if a volume_key was specified
                volumes = None
                if self.volume_key:
                    with h5py.File(self.h5_path, "r") as f:
                        g = f["epochs"][f"epoch_{ep}"]
                        if self.volume_key in g:
                            volumes = g[self.volume_key][:]
                        else:
                            print(f"  ⚠️ Volume key '{self.volume_key}' not found in epoch {ep}. Using empirical counts.")

                # Compute point counts using optimized routing
                df_counts = self._compute_region_counts(tree, points, labels, unique_classes, volumes)
                
                # Directly compute Mutual Information (KL Divergence)
                if not df_counts.empty:
                    mi_df = self._compute_mutual_information(df_counts, N_total, unique_classes)
                    mi_df.insert(0, "epoch", ep)
                    self.estimates["Mutual Information"].append(mi_df)
                    
            except Exception as e:
                print(f"  ⚠️ Skipping epoch {ep}: {e}")

        # 3. Concatenate all epochs into unified DataFrames
        for key, frames in self.estimates.items():
            if frames:
                self.estimates[key] = (
                    pd.concat(frames, ignore_index=True)
                    .rename_axis(None, axis=1)
                )

    def _compute_region_counts(self, tree: Tree, points: np.ndarray, labels: np.ndarray, unique_classes: np.ndarray, volumes: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Uses the Tree's optimized perform_number_count to route points 
        and formats the output, appending volume data if provided.
        """
        all_layer_data = []
        
        # 1. Call the highly optimized routing engine
        counts_dict = tree.perform_number_count(points, y=labels)
        
        # 2. Reformat the nested dictionary into a list of row dictionaries
        for layer_idx, regions in counts_dict.items():
            if layer_idx == 0:
                continue 
                
            for r_id, class_counts in regions.items():
                row_data = {
                    "layer_idx": layer_idx,
                    "region_idx": r_id,
                }
                
                total_in_region = 0
                for cls in unique_classes:
                    count = class_counts.get(cls, 0)
                    row_data[str(int(cls))] = count
                    total_in_region += count
                    
                row_data["total"] = total_in_region
                
                if volumes is not None and r_id < len(volumes):
                    row_data["volume"] = volumes[r_id]
                
                # Only append regions that actually contain points
                if total_in_region > 0:
                    all_layer_data.append(row_data)

        return pd.DataFrame(all_layer_data)

    def _compute_mutual_information(self, df_counts: pd.DataFrame, N: int, unique_classes: np.ndarray) -> pd.DataFrame:
        """
        Computes the Mutual Information I(Y; W) layer by layer.
        I(Y; W) = sum_{y, w} P(y, w) * log(P(y, w) / (P(y) * P(w)))
        """
        class_cols = [str(int(c)) for c in unique_classes]
        results = {}
        
        has_volumes = "volume" in df_counts.columns
        
        for layer_idx, group in df_counts.groupby("layer_idx"):
            
            if has_volumes:
                # ---------------------------------------------------------
                # VOLUME-BASED PROBABILITIES (Bounded [0,1]^n Space)
                # ---------------------------------------------------------
                V_w = group["volume"].to_numpy()
                
                # Normalize so probabilities sum to 1 over the populated regions.
                # If no regions are empty, V_w.sum() will inherently be 1.0.
                P_w = V_w / V_w.sum() if V_w.sum() > 0 else np.zeros_like(V_w)
                
                # P(Y|W) remains the empirical ratio within the region: N_{y,w} / N_w
                P_y_given_w = group[class_cols].to_numpy() / group["total"].to_numpy()[:, None]
                
                # Joint P(Y, W) = P(Y|W) * P_w
                P_yw = P_y_given_w * P_w[:, None]
                
            else:
                # ---------------------------------------------------------
                # EMPIRICAL PROBABILITIES (Standard)
                # ---------------------------------------------------------
                # Joint probability P(Y=y, W=w) based strictly on points
                P_yw = group[class_cols].to_numpy() / N
                
                # Marginal probability P(W=w) based strictly on points
                P_w = group["total"].to_numpy() / N
            
            # Marginal probability P(Y=y)
            P_y = P_yw.sum(axis=0)
            
            # Product of marginals P(Y=y) * P(W=w)
            denominator = P_w[:, None] * P_y[None, :]
            
            # Mask out zeros to avoid log(0)
            mask = P_yw > 0
            
            kl_terms = np.zeros_like(P_yw)
            kl_terms[mask] = P_yw[mask] * np.log(P_yw[mask] / denominator[mask])
            
            mi = kl_terms.sum()
            results[f"l{layer_idx}"] = mi
            
        return pd.DataFrame([results])

    def get_estimates(self) -> Dict[str, pd.DataFrame]:
        return self.estimates