import pathlib as pl

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage
data_storage = current_dir/"stored_data"

# New storage facility
new_data_storage = current_dir/"new_stored_data"

testing_data = data_storage/"testing"

## Data paths
def moon_path(arch, dropout, noise, run_number):
    return data_storage/"moon"/f"arch_{arch}"/f"dropout_{dropout:.2f}"/f"noise_{noise:.2f}"/f"run_{int(run_number)}"

def wbc_path(arch, dropout, noise, run_number):
    return data_storage / "wbc" / f"arch_{arch}" / f"dropout_{dropout:.2f}" / f"noise_{noise:.2f}" / f"run_{int(run_number)}"


def get_new_path(dataset, noise, dropout, run_number):
    assert dataset in ["moons", "wbc", "wine", "hd", "car"], "Invalid dataset"
    assert noise in [0.0, 0.2, 0.4], "Invalid noise"
    assert dropout in [0.0, 0.1, 0.3, 0.5], "Invalid dropout"
    assert run_number in [1,2,3,4,5], "Invalid run number"
    return_path = new_data_storage / dataset / f"noise_{noise:.2f}" / f"dropout_{dropout:.2f}" / f"run_{int(run_number)}"
    return return_path

def get_test_data():
    return testing_data