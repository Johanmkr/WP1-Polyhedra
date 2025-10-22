import numpy as np 
import matplotlib.pyplot as plt 
import src_experiment as scf
import geobin as gb 
import pathlib as pl
import torch
from sklearn.datasets import make_moons


training_seed = 42
testing_seed = 41
inference_seed = 40



# Read config file and/or parse cmb-line arguments
args = scf.get_args()



# Check if the experiment path exists, if not, create 
experiment_path = scf.get_path_to_experiment_storage(args.experiment_name)
if not experiment_path.exists():
    createfolders(experiment_path)


# Make the data
train_data = scf.make_moon_dataloader()

# Set up and train the model
from IPython import embed; embed()


if __name__ == "__main__":
    for arg in vars(args):
        print(arg, getattr(args, arg)) 
