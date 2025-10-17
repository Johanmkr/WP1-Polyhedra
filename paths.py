import pathlib as pl 

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage

data_storage = current_dir/"store_data"

def get_path_to_experiment_storage(experiment_name):
    ###TODO
    # Add check if dir exists or not
    return data_storage/experiment_name


