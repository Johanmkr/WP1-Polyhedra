import pathlib as pl 

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage
data_storage = current_dir/"stored_data"

testing_data = data_storage/"testing"

moon_experiment_storage = data_storage/"variable_noise_moons_experiment"

def get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number):
    # Check if input is valid
    assert model_name in ["small_uniform", "medium_uniform", "large_uniform", "increasing", "decreasing"], f"Model name {model_name} not recognized. Must be one of ['small_uniform', 'medium_uniform', 'large_uniform', 'increasing', 'decreasing']."
    assert dataset_name in ["small", "medium", "large"], f"Dataset name {dataset_name} not recognized. Must be one of ['small', 'medium', 'large']."
    assert 0.0 <= noise_level <= 1.0, f"Noise level {noise_level} not in range [0.0, 1.0]."
    assert isinstance(run_number, int) and 25 >= run_number > 0, f"Run number {run_number} must be a non-negative integer in the range [0, 25]."
        # Format the numpers

    return moon_experiment_storage/f"model_{model_name}"/f"dataset_{dataset_name}"/f"noise_{noise_level:.1f}"/f"run_{int(run_number)}"





## TEST MOON DATA STORAGE
test_moon_storage = data_storage/"test_moon"
def get_test_moon_path(model_name, dataset_name, noise_level, run_number):
    return test_moon_storage/f"model_{model_name}"/f"dataset_{dataset_name}"/f"noise_{noise_level:.1f}"/f"run_{int(run_number)}"


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


