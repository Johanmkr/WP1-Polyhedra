import numpy as np
import torch.nn as nn
from sklearn.datasets import make_moons
from pathlib import Path
import torch
import pandas as pd
import os
from .paths import get_path_to_moon_experiment_storage

# from . import functions


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.hidden_sizes = hidden_sizes

        for i in range(len(hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            if i == 0:
                setattr(self, layer_name, nn.Linear(input_size, hidden_sizes[i]))
                setattr(self, relu_name, nn.ReLU())
            else:
                setattr(
                    self, layer_name, nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
                )
                setattr(self, relu_name, nn.ReLU())

        output_layer_name = f"l{len(hidden_sizes) + 1}"

        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))
        
    def forward(self, x):
        out = x
        for i in range(len(self.hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            out = getattr(self, layer_name)(out)
            out = getattr(self, relu_name)(out)

        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        out = getattr(self, output_layer_name)(out)
        # out = getattr(self, "output_activation")(out)
        # if len(out.shape) == 1:
        #     out = out.unsqueeze(0)
        # elif len(out.shape) == 2 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        # elif len(out.shape) == 3 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        return out

def createfolders(*dirs: Path) -> None:
    """
    Create folders for storing data
    """
    for dir in dirs:
        dir.mkdir(parents=True, exist_ok=True)
        
def get_specific_moon_state_dict(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    run_number: int,
    epoch: int,
    ) -> Path:
    assert isinstance(epoch, int) and epoch >= 0, f"Epoch {epoch} must be a non-negative integer."
    base_path = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
    state_dict_path = base_path/"state_dicts"/f"epoch{epoch}.pth"
    
    state_dict = torch.load(state_dict_path)
    return state_dict

def get_df_of_run_summary(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    run_number: int,
):

    base_path = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
    summary_path = base_path/"run_summary.csv"
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Run summary file not found at {summary_path}")
    
    df = pd.read_csv(summary_path)
    return df

def get_df_of_convergence_summary(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    run_number: int,
):

    base_path = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
    summary_path = base_path/"convergence_summary.csv"
    
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Convergence summary file not found at {summary_path}")
    
    df = pd.read_csv(summary_path)
    return df




if __name__ == "__main__":
    pass
