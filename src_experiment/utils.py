from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


import torch
import torch.nn as nn
import torch.nn.init as init
from typing import List, Optional


class NeuralNet(nn.Module):
    """
    Fully-connected feedforward neural network with ReLU activations
    and Dropout regularization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        dropout: float = 0.0,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout  

        # ------------------------------------------------------------------
        # Hidden Layers
        # ------------------------------------------------------------------
        for i, hidden_dim in enumerate(hidden_sizes):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            dropout_name = f"dropout{i + 1}"

            in_features = input_size if i == 0 else hidden_sizes[i - 1]

            setattr(self, layer_name, nn.Linear(in_features, hidden_dim))
            setattr(self, relu_name, nn.ReLU())
            setattr(self, dropout_name, nn.Dropout(p=dropout))

        # ------------------------------------------------------------------
        # Output Layer
        # ------------------------------------------------------------------
        output_layer_name = f"l{len(hidden_sizes) + 1}"
        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))

        if seed is not None:
            self.set_seed(seed)

    # ----------------------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x

        for i in range(len(self.hidden_sizes)):
            out = getattr(self, f"l{i + 1}")(out)
            out = getattr(self, f"relu{i + 1}")(out)
            out = getattr(self, f"dropout{i + 1}")(out)

        out = getattr(self, f"l{len(self.hidden_sizes) + 1}")(out)
        return out

    # ----------------------------------------------------------------------
    # Initialization Utilities
    # ----------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize hidden layers
        for i in range(len(self.hidden_sizes)):
            layer = getattr(self, f"l{i + 1}")
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        # Initialize output layer
        output_layer = getattr(self, f"l{len(self.hidden_sizes) + 1}")
        if isinstance(output_layer, nn.Linear):
            init.xavier_uniform_(output_layer.weight)
            if output_layer.bias is not None:
                init.zeros_(output_layer.bias)

# ----------------------------------------------------------------------
# Filesystem Utilities
# ----------------------------------------------------------------------

def createfolders(*dirs: Path) -> None:
    """
    Create folders for storing data.

    Parameters
    ----------
    *dirs : pathlib.Path
        One or more directory paths to create.
    """
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    pass
