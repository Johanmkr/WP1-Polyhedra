import numpy as np 
import matplotlib.pyplot as plt 
from arg_parser import get_args, createfolders
import geobin as gb 
import pathlib as pl

# Read config file and/or parse cmb-line arguments
args = get_args()

# Make sure the paths used exists and are valid
current_path = pl.Path(__file__).parent.resolve()
datapath = current_path / "stored_data"

# Check if the experiment path exists, if not, create 
experiment_path = datapath / args.experiment_name
if not experiment_path.exists():
    createfolders(experiment_path)

# Set up and train the model
from IPython import embed; embed()


if __name__ == "__main__":
    for arg in vars(args):
        print(arg, getattr(args, arg)) 
