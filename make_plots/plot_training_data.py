import numpy as np 
import matplotlib.pyplot as plt 
import pathlib as pl 
import sys
import pandas as pd
sys.path.append(str(pl.Path(__file__).resolve().parent.parent))

from src_experiment import paths

def plot_trainin_data(exp_name:str = "default_experiment") -> None:
     filename = paths.get_path_to_experiment_storage(exp_name) / f"{exp_name}_training_data.pkl"
     training_data = pd.read_pickle(filename)
     fig, ax = plt.subplots()
     axy = ax.twinx()
     
     ax.plot(training_data["train_loss"], ls="--", color="blue", label="Train loss")
     ax.plot(training_data["test_loss"], color="blue", label="Test loss")
     axy.plot(training_data["train_accuracy"], ls="--", color="red", label="Train accuracy")
     axy.plot(training_data["test_accuracy"], color="red", label="Test accuracy")
     ax.set_xlabel("Epochs")
     ax.set_ylabel("Loss")
     axy.set_ylabel("Accuracy")
     ax.legend()
     axy.legend()
     plt.show()


if __name__ == "__main__":
   plot_trainin_data() 
