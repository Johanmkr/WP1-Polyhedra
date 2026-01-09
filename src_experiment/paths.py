import pathlib as pl 

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage
data_storage = current_dir/"stored_data"

testing_data = data_storage/"testing"

## TEST MOON DATA STORAGE
test_moon_storage = data_storage/"test_moon"
def get_test_moon_path(model_name, dataset_name, noise_level, run_number):
    return test_moon_storage/f"model_{model_name}"/f"dataset_{dataset_name}"/f"noise_{noise_level:.2f}"/f"run_{int(run_number)}"

wbc_storage = data_storage / "wbc"
def get_wbc_storage(model_name, run_number):
    return wbc_storage / f"model_{model_name}" / f"run_{int(run_number)}"


def get_storage_path(type="moons", **kwargs):
    match type:
        case "moons":
            return get_test_moon_path(**kwargs)
        case "wbc":
            return get_wbc_storage(**kwargs)


