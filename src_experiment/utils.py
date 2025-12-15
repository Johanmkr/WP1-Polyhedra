import numpy as np
from sklearn.datasets import make_moons
from pathlib import Path
import torch
import pandas as pd
import os
if __name__ == "__main__":
    from paths import get_path_to_moon_experiment_storage
 
else:
    from .paths import get_path_to_moon_experiment_storage


# from . import functions


import torch
import torch.nn as nn
import torch.nn.init as init

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, seed: int = None):
        super(NeuralNet, self).__init__()
        self.hidden_sizes = hidden_sizes

        # Create hidden layers dynamically
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

        # Create output layer
        output_layer_name = f"l{len(hidden_sizes) + 1}"
        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))
        
        # If seed is provided, set it and initialize weights
        if seed is not None:
            self.set_seed(seed)
        
    def forward(self, x):
        out = x
        for i in range(len(self.hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            out = getattr(self, layer_name)(out)
            out = getattr(self, relu_name)(out)

        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        out = getattr(self, output_layer_name)(out)
        return out

    def set_seed(self, seed: int):
        """Set random seed and initialize all network weights."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Initialize hidden layers
        for i in range(len(self.hidden_sizes)):
            layer = getattr(self, f"l{i + 1}")
            if isinstance(layer, nn.Linear):
                init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    init.zeros_(layer.bias)

        # Initialize output layer
        output_layer = getattr(self, f"l{len(self.hidden_sizes) + 1}")
        if isinstance(output_layer, nn.Linear):
            init.xavier_uniform_(output_layer.weight)
            if output_layer.bias is not None:
                init.zeros_(output_layer.bias)


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


def delete_tree_pkl_from_run(model_name, dataset_name, noise_level, run_number):
    base_path = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
    tree_name = base_path / "trees.pkl"
    if os.path.exists(tree_name) and str(tree_name)[-4:] == ".pkl": # Check if the path exists an is a pickle file
        os.remove(tree_name)
    else:
        print("File does not exist")
        
def remove_all_tree_pickle_files():
    model_names = ["small_uniform", "medium_uniform", "large_uniform", "decreasing", "increasing"]
    dataset_names = ["small", "medium", "large"]
    noises = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    run_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    for model in model_names:
        for dataset in dataset_names:
            for noise in noises:
                for run_number in run_numbers:
                    delete_tree_pkl_from_run(model, dataset, noise, run_number)

if __name__ == "__main__":
    pass