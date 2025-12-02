"""Basic outline of script

1. Load datasets and models. 
2. For each model, dataset and noise level:
    0. Create the folder to store the state dicts if it does not exist.
    a. Train the model on the training data.
    b. Evaluate the model on the test data.
    c. Check if the model converges
    e. Save the state dicts if it converges. 
    
savepath must have the following structure:

./stored_data/
    experiment_name/
        model_name/
            dataset_name/
                noise_level/
                    randomized_run_nr/
                        state_dicts/
                            somestate.pth   || This is the savepath argument
                        run_summary.csv
                        
                    
model_names = ["small_uniform", "small_uniform_long", "medium_uniform", "large_uniform", "increasing", "decreasing"]

dataset_names = ["small", "medium", "large"]

noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
"""


# Import necessary libraries
from src_experiment import get_args, createfolders, moons_models, train_model, datasets
from src_experiment.paths import data_storage

# GLOABL PARAMETERS
SAVE_STATES = True
RETURN_STATES = False
save_everyth_epoch = 10
EPOCHS = 250
EXPERIMENT_NAME = "variable_noise_moons_experiment"
nr_sample_runs = 5

def train_single_model_on_moons(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    randomized_run_nr: int = 0
):
    # Create savepath
    savepath = data_storage/EXPERIMENT_NAME/model_name/dataset_name/f"noise_{noise_level}"/f"run_{randomized_run_nr}"/"state_dicts"
    createfolders(savepath)
    
    # Get model
    model = moons_models[model_name]
    
    # Get data
    data_loaders = datasets[dataset_name][noise_level]
    train_loader = data_loaders["train"]
    test_loader = data_loaders["test"]
    
    # Train model
    results = train_model(
        model=model,
        train_data=train_loader,
        test_data=test_loader,
        epochs=EPOCHS,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=save_everyth_epoch
    )
    
def main():
    for model_name in moons_models.keys():
        for dataset_name in datasets.keys():
            for noise_level in datasets[dataset_name].keys():
                i = 0
                while i < nr_sample_runs:
                    print(f"Training {model_name} on {dataset_name} moons with noise level {noise_level}")
                    train_single_model_on_moons(
                        model_name=model_name,
                        dataset_name=dataset_name,
                        noise_level=noise_level,
                        randomized_run_nr=i
                    )
                    i += 1


if __name__ == "__main__":
    train_single_model_on_moons(
        model_name="small_uniform",
        dataset_name="small",
        noise_level=0.0
    )