from __future__ import annotations
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Iterable
import pickle
from .region_tree import RegionTree
from .tree_node import TreeNode
from src_experiment import get_path_to_moon_experiment_storage, get_test_moon_path

QUANTITIES_TO_ESTIMATE = ["MI_KL", "MI_IS", "MI_D"]


class EstimateQuantities1Run:
    """
    Estimate information-theoretic quantities for a single training run of a model.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset.
    noise_level : float
        Noise level applied to the dataset.
    run_number : int
        Identifier for the run.
    calculate : bool, optional
        If True, automatically calculate estimates upon initialization.
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
        self.noise_level = noise_level
        self.run_number = run_number

        # Paths to data
        self.data_dir = get_test_moon_path(model_name, dataset_name, noise_level, run_number)
        self.data_path = self.data_dir / "number_counts_per_epoch.pkl"

        # Load number counts
        self.ncounts: Dict[int, pd.DataFrame] = self._open_object(self.data_path)
        self.epochs: np.ndarray = np.array(list(self.ncounts.keys()))

        # Initialize estimates dictionary
        self.estimates: Dict[str, List[pd.DataFrame]] = {q: [] for q in QUANTITIES_TO_ESTIMATE}

        if calculate:
            self.calculate_estimates()

    # ----------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------

    def get_estimates(self) -> Dict[str, List[pd.DataFrame]]:
        """Return the calculated estimates."""
        return self.estimates

    def calculate_estimates(self) -> None:
        """
        Calculate information-theoretic quantities for all epochs.
        
        Procedure
        ---------
        1. Sort region counts by layer and region indices.
        2. Run consistency checks.
        3. Compute probability masses and estimates per layer.
        4. Aggregate layer-wise results into final DataFrame.
        """
        for epoch, frame in self.ncounts.items():
            frame = frame.sort_values(by=["layer_idx", "region_idx"]).reset_index(drop=True)
            frame, num_layers, classes, n_k = self._run_consistency_check_on_single_frame(frame)

            # Extract counts
            n_w = np.array(frame['total'])
            n_kw = np.array(frame[classes])
            N = n_k.sum()

            # Probability masses
            m_w = np.expand_dims(n_w / N, axis=1)
            m_kw = n_kw / N
            m_k = n_k / N

            # Compute estimates
            for estimate_name, results in self.estimates.items():
                frame[estimate_name] = self._individual_estimates(estimate_name, m_w, m_kw, m_k, N)
                result = (
                    frame.groupby('layer_idx')[estimate_name].sum()
                    .rename(lambda i: f"l{i}")
                    .to_frame()
                    .T
                )
                result.insert(0, "epoch", epoch)
                result.index.name = None
                results.append(result)

        # Aggregate layer-wise results
        for estimate_name, results in self.estimates.items():
            newframe = pd.concat(results, ignore_index=True)
            newframe = newframe.rename_axis(None, axis=1)
            self.estimates[estimate_name] = newframe

    # ----------------------------------------------------------------------
    # Internal Methods
    # ----------------------------------------------------------------------

    def _individual_estimates(
        self,
        estimate: str,
        m_w: np.ndarray,
        m_kw: np.ndarray,
        m_k: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """Compute a single estimate type for all regions."""
        match estimate:
            case "MI_KL":
                logterm = m_kw / (m_w @ m_k)
                logterm = np.log(logterm, where=logterm > 0)
                return (m_kw * logterm).sum(axis=1)
            case "MI_IS":
                term = m_kw / (m_w @ m_k)
                logterm = np.log(term, where=term > 0)
                return (term - logterm).sum(axis=1)
            case "MI_D":
                return np.zeros(m_w.shape[0])
            case _:
                return np.zeros(m_w.shape[0])

    def _run_consistency_check_on_single_frame(
        self, frame: pd.DataFrame
    ) -> Tuple[pd.DataFrame, int, List[str], np.ndarray]:
        """
        Perform consistency checks on region count DataFrame.
        
        Returns
        -------
        frame : pd.DataFrame
            Sorted and checked frame.
        num_layers : int
            Number of layers.
        classes : list[str]
            List of class labels.
        n_k : np.ndarray
            Total counts per class.
        """
        assert frame.columns[-1] == "total", "Last column should be 'total'"

        classes = frame.columns[2:-1].tolist()
        row_sums = frame[classes].sum(axis=1)
        totals = frame["total"]
        assert all(row_sums == totals), "Row sums of class counts should equal total counts"

        num_layers = frame["layer_idx"].nunique()
        total_counts_per_layer = frame.groupby("layer_idx")["total"].sum().values
        assert all(total_counts_per_layer == total_counts_per_layer[0]), "Total counts should be the same for each layer"

        # Ensure class division is consistent across layers
        n_k1 = frame.loc[frame["layer_idx"] == 1, classes].sum(axis=0).to_numpy()[None, :]
        for layer in range(1, num_layers + 1):
            n_k = frame.loc[frame["layer_idx"] == layer, classes].sum(axis=0).to_numpy()[None, :]
            assert np.all(n_k1 == n_k)

        return frame, num_layers, classes, n_k

    def _open_object(self, filename: str):
        """Open a pickled object."""
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        return obj


class AverageEstimates:
    """
    Collect and aggregate estimates across multiple runs.
    
    Parameters
    ----------
    model_name : str
        Name of the model.
    dataset_name : str
        Name of the dataset.
    noise_level : float
        Noise level applied to the dataset.
    """

    def __init__(
        self,
        model_name: str = "small",
        dataset_name: str = "new",
        noise_level: float = 0.05,
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.noise_level = noise_level
        self.run_numbers: np.ndarray = np.arange(35)

    # ----------------------------------------------------------------------
    # Public Methods
    # ----------------------------------------------------------------------

    def _find_from_all_runs(self) -> None:
        """
        Collect estimates for all runs and store in a dictionary.
        """
        self.individual_estimates: Dict[str, List[pd.DataFrame]] = {q: [] for q in QUANTITIES_TO_ESTIMATE}

        for run_number in self.run_numbers:
            run = EstimateQuantities1Run(
                model_name=self.model_name,
                dataset_name=self.dataset_name,
                noise_level=self.noise_level,
                run_number=int(run_number)
            )
            run.calculate_estimates()
            for key, val in run.get_estimates().items():
                self.individual_estimates[key].append(val)


if __name__ == "__main__":
    pass
