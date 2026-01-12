import pathlib as pl

# Get path to the parent directory of this file
current_dir = pl.Path(__file__).parent.parent.resolve()

# Get path to current working directory
working_dir = pl.Path().resolve()

# Path to the data storage
data_storage = current_dir/"stored_data"

testing_data = data_storage/"testing"

## Data paths
def moon_path(arch, dropout, noise, run_number):
    return data_storage/"moon"/f"arch_{arch}"/f"dropout_{dropout:.2f}"/f"noise_{noise:.2f}"/f"run_{int(run_number)}"

def wbc_path(arch, dropout, noise, run_number):
    return data_storage / "wbc" / f"arch_{arch}" / f"dropout_{dropout:.2f}" / f"noise_{noise:.2f}" / f"run_{int(run_number)}"
