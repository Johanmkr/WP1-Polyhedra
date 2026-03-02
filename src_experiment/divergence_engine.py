from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

QUANTITIES_TO_ESTIMATE = ["Kullback-Leibler", "Itakura-Saito", "H(W)", "H(Y|W)"]

class DivergenceEngine:
    """
    General-purpose engine for computing information-theoretic
    divergences from region-wise number counts.
    """

    def __init__(self, frame: pd.DataFrame):
        """
        Parameters
        ----------
        frame : pd.DataFrame
            DataFrame with columns:
            ['layer_idx', 'region_idx', class_1, ..., class_K, 'total']
        """
        self.frame = (
            frame.sort_values(by=["layer_idx", "region_idx"])
            .reset_index(drop=True)
        )

        (
            self.frame,
            self.num_layers,
            self.classes,
            self.n_k,
        ) = self._run_consistency_check(self.frame)

        self.N = int(self.n_k.sum())

        # Precompute probability masses
        self._prepare_probability_masses()

    def compute(self) -> Dict[str, pd.DataFrame]:
        """
        Compute all divergences and return layer-wise aggregates.
        """
        results = {}

        for quantity in QUANTITIES_TO_ESTIMATE:
            values = self._compute_individual(quantity)
            self.frame[quantity] = values

            layerwise = (
                self.frame
                .groupby("layer_idx")[quantity]
                .sum()
                .rename(lambda i: f"l{i}")
                .to_frame()
                .T
            )

            layerwise.index.name = None
            results[quantity] = layerwise

        return results

    def _prepare_probability_masses(self) -> None:
        n_w = self.frame["total"].to_numpy()
        n_kw = self.frame[self.classes].to_numpy()

        self.m_w = np.expand_dims(n_w / self.N, axis=1)
        self.m_kw = n_kw / self.N
        self.m_k = self.n_k / self.N

    def _compute_individual(self, estimate: str) -> np.ndarray:
        match estimate:
            case "Kullback-Leibler":
                logterm = self.m_kw / (self.m_w @ self.m_k)
                outterm = np.full_like(logterm, 0, dtype=float)
                logterm = np.log(logterm, where=logterm > 0, out=outterm)
                return (self.m_kw * logterm).sum(axis=1)

            case "Itakura-Saito":
                term = self.m_kw / (self.m_w @ self.m_k)
                outterm = np.full_like(term, 0, dtype=float)
                logterm = np.log(term, where=term > 0, out=outterm)
                return (term - logterm - 1).sum(axis=1)

            case "H(W)":
                logterm = np.log(self.m_w, where=self.m_w > 0, out=np.zeros_like(self.m_w, dtype=float))
                return - (self.m_w * logterm).squeeze()
            
            case "H(Y|W)":
                term = self.m_kw / self.m_w
                logterm = np.log(term, where=term > 0, out=np.zeros_like(term, dtype=float))
                return - (self.m_kw * logterm).sum(axis=1)

            case _:
                raise ValueError(f"Unknown estimate: {estimate}")

    def _run_consistency_check(self, frame: pd.DataFrame) -> Tuple[pd.DataFrame, int, List[str], np.ndarray]:
        assert frame.columns[-1] == "total", "Last column must be 'total'"
        classes = frame.columns[2:-1].tolist()

        # Row consistency (Sum of classes = total points in region)
        assert np.all(frame[classes].sum(axis=1) == frame["total"])

        # Layer consistency (Sum of points in layer = N)
        num_layers = frame["layer_idx"].nunique()
        totals_per_layer = frame.groupby("layer_idx")["total"].sum().values
        assert np.all(totals_per_layer == totals_per_layer[0]), "Point leak! A layer has fewer/more points than others."

        # Class totals consistency (Sum of class points across regions matches global distribution)
        n_k_ref = (
            frame.loc[frame["layer_idx"] == 1, classes]
            .sum(axis=0)
            .to_numpy()[None, :]
        )

        for layer in range(1, num_layers + 1):
            n_k = (
                frame.loc[frame["layer_idx"] == layer, classes]
                .sum(axis=0)
                .to_numpy()[None, :]
            )
            assert np.all(n_k == n_k_ref), f"Class marginal mismatch at layer {layer}."

        return frame, num_layers, classes, n_k_ref