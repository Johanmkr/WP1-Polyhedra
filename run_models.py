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
state_dict_path = experiment_path/"state_dicts"
if not state_dict_path.exists():
    scf.createfolders(state_dict_path)

# Make the data
train_data = scf.make_moon_dataloader(n_samples=10000, noise=0.15, random_state=training_seed, batch_size=250)
test_data = scf.make_moon_dataloader(n_samples=500, noise=0.15, random_state=testing_seed, batch_size=250)

# Set up the model based on the parsed arguments or the config file
model = scf.NeuralNet(input_size=2, hidden_sizes=args.architecture, num_classes=1)

# Train the model on the training data
scf.train_model(model, train_data, test_data, args.epochs, state_dict_path, args.experiment_name, SAVE_STATES=True, save_everyth_epoch=10)



# Set up and train the model
#from IPython import embed; embed()


if __name__ == "__main__":
    for arg in vars(args):
        print(arg, getattr(args, arg)) 
