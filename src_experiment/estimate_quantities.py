from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List

# from src_experiment import (
#     get_storage_path,
# )

from .divergence_engine import (
    DivergenceEngine,
    QUANTITIES_TO_ESTIMATE,
)


class EstimateQuantities1Run:
    """
    Experiment-specific wrapper that:
    - loads number counts
    - loops over epochs
    - delegates all math to DivergenceEngine
    """

    def __init__(
        self,
        data_dir
    ):
        self.data_dir = data_dir
        self.data_path = self.data_dir / "number_counts_per_epoch.pkl"
        self.ncounts: Dict[int, pd.DataFrame] = self._open_object(self.data_path)

        self.estimates: Dict[str, List[pd.DataFrame]] = {
            q: [] for q in QUANTITIES_TO_ESTIMATE
        }

        # Perform calculations
        self.calculate_estimates()

    # ------------------------------------------------------------------

    def calculate_estimates(self) -> None:
        for epoch, frame in self.ncounts.items():
            engine = DivergenceEngine(frame)
            epoch_results = engine.compute()

            for key, df in epoch_results.items():
                df.insert(0, "epoch", epoch)
                self.estimates[key].append(df)

        # Concatenate epochs
        for key, frames in self.estimates.items():
            self.estimates[key] = (
                pd.concat(frames, ignore_index=True)
                .rename_axis(None, axis=1)
            )

    def get_estimates(self) -> Dict[str, pd.DataFrame]:
        return self.estimates

    # ------------------------------------------------------------------

    @staticmethod
    def _open_object(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)


class AverageEstimates:
    """
    Higher-level operator aggregating over runs.
    """

    def __init__(
        self,
        data_dirs: List,
    ):
        self.data_dirs = data_dirs
        self.means = {q: [] for q in QUANTITIES_TO_ESTIMATE}
        self.stds = {q: [] for q in QUANTITIES_TO_ESTIMATE}
        self._find_means_and_stds()

    def _collect(self) -> Dict[str, List[pd.DataFrame]]:
        aggregated = {q: [] for q in QUANTITIES_TO_ESTIMATE}

        for data_dir in self.data_dirs:
            run = EstimateQuantities1Run(
                data_dir=data_dir
            )

            for key, df in run.get_estimates().items():
                aggregated[key].append(df)

        return aggregated
    
    def _concatenate(self) -> Dict[str, pd.DataFrame]:
        aggregated = self._collect()
        concatenated = {}

        for key, dfs in aggregated.items():
            concatenated[key] = pd.concat(dfs, keys=np.arange(len(dfs)), names=["run", "row"]).reset_index(level="run")

        return concatenated
    
    def _find_means_and_stds(self) -> None:
        concatenated = self._concatenate()

        for key, df in concatenated.items():
            summary = (df
                       .groupby("epoch")[[col for col in df.columns[1:] if col not in ["run", "epoch"]]]
                       .agg(["mean", "std"])
                       )

            self.means[key] = summary.xs("mean", axis=1, level=1).reset_index()
            self.stds[key] = summary.xs("std", axis=1, level=1).reset_index()
