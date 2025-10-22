import pathlib as pl 

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage
data_storage = working_dir/"stored_data"


# Desired datastructure
# - stored_data
# | - experiment_name
# | | - state_dicts
# | | | - experiment_name_epoch1.pth
# | | | - experiment_name_epoch2.pth
# | | | - experiment_name_epoch3.pth
# | | | - ...
# | | - training_data.pkl
# | | - experiment_name_configuration.yaml
def get_path_to_experiment_storage(experiment_name):
    ###TODO
    # Add check if dir exists or not
    return data_storage/experiment_name


