from __future__ import annotations
from pathlib import Path
from typing import Iterable, List, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


class NeuralNet(nn.Module):
    """
    Fully-connected feedforward neural network with ReLU activations.

    Hidden layers are created dynamically based on the provided
    `hidden_sizes` list. Layer names follow the convention:

        l1, relu1, l2, relu2, ..., lK, reluK, l(K+1)

    where l(K+1) is the output layer.

    Parameters
    ----------
    input_size : int
        Dimensionality of the input.
    hidden_sizes : list[int]
        Number of neurons in each hidden layer.
    num_classes : int
        Number of output classes.
    seed : int, optional
        Random seed for reproducible weight initialization.
    """

    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        num_classes: int,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hidden_sizes = hidden_sizes

        # ------------------------------------------------------------------
        # Hidden Layers
        # ------------------------------------------------------------------

        for i, hidden_dim in enumerate(hidden_sizes):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"

            in_features = input_size if i == 0 else hidden_sizes[i - 1]
            setattr(self, layer_name, nn.Linear(in_features, hidden_dim))
            setattr(self, relu_name, nn.ReLU())

        # ------------------------------------------------------------------
        # Output Layer
        # ------------------------------------------------------------------

        output_layer_name = f"l{len(hidden_sizes) + 1}"
        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))

        # Optional seeded initialization
        if seed is not None:
            self.set_seed(seed)

    # ----------------------------------------------------------------------
    # Forward Pass
    # ----------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, input_size).

        Returns
        -------
        torch.Tensor
            Output logits of shape (batch_size, num_classes).
        """
        out = x

        for i in range(len(self.hidden_sizes)):
            out = getattr(self, f"l{i + 1}")(out)
            out = getattr(self, f"relu{i + 1}")(out)

        out = getattr(self, f"l{len(self.hidden_sizes) + 1}")(out)
        return out

    # ----------------------------------------------------------------------
    # Initialization Utilities
    # ----------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        """
        Set random seed and initialize all network weights.

        Hidden layers use Kaiming initialization (ReLU),
        output layer uses Xavier initialization.
        """
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
