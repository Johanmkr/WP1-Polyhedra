from __future__ import annotations
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List

from src_experiment import (
    get_test_moon_path,
)

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
        model_name: str = "small",
        dataset_name: str = "new",
        noise_level: float = 0.05,
        run_number: int = 0,
        calculate: bool = False,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.noise_level = noise_level
        self.run_number = run_number

        self.data_dir = get_test_moon_path(
            model_name,
            dataset_name,
            noise_level,
            run_number,
        )

        self.data_path = self.data_dir / "number_counts_per_epoch.pkl"
        self.ncounts: Dict[int, pd.DataFrame] = self._open_object(self.data_path)

        self.estimates: Dict[str, List[pd.DataFrame]] = {
            q: [] for q in QUANTITIES_TO_ESTIMATE
        }

        if calculate:
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
        model_name: str = "small",
        dataset_name: str = "new",
        noise_level: float = 0.05,
        run_numbers: np.ndarray | None = None,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.noise_level = noise_level
        self.run_numbers = run_numbers if run_numbers is not None else np.arange(35)

    def collect(self) -> Dict[str, List[pd.DataFrame]]:
        aggregated = {q: [] for q in QUANTITIES_TO_ESTIMATE}

        for run_number in self.run_numbers:
            run = EstimateQuantities1Run(
                model_name=self.model_name,
                dataset_name=self.dataset_name,
                noise_level=self.noise_level,
                run_number=int(run_number),
                calculate=True,
            )

            for key, df in run.get_estimates().items():
                aggregated[key].append(df)

        return aggregated
