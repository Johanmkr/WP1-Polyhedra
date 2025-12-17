"""Basic outline of script

1. Load datasets and models. 
2. For each model, dataset and noise level:
    0. Create the folder to store the state dicts if it does not exist. This is done by the training loop only if the model should be saved. 
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
from tqdm import trange
import numpy as np
from src_experiment import train_model, get_new_moons_data_for_all_noises, get_model
from src_experiment.paths import get_test_moon_path

# GLOABL PARAMETERS
SAVE_STATES = True
RETURN_STATES = False
save_everyth_epoch = 5
EPOCHS = 125

train_data = get_new_moons_data_for_all_noises(type="training")
test_data = get_new_moons_data_for_all_noises(type="testing")


def train_single_model_on_moons(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    run_number: int = 0
):
    # Create savepath
    savepath = get_test_moon_path(model_name=model_name, dataset_name=dataset_name, noise_level=noise_level, run_number=run_number)
    
    # Get data
    train_loader = train_data[noise_level]
    test_loader = test_data[noise_level]
    
    
    # Train model
    run_results = train_model(
        model=get_model(model_name, seed=run_number),
        train_data=train_loader,
        test_data=test_loader,
        epochs=EPOCHS,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=save_everyth_epoch,
    )
    
    return run_results
    
def main():
    for noise in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for run_number in np.arange(35):
            print(f"Noise: {noise}\nRun {run_number}/{np.arange(35).max()}")
            train_single_model_on_moons("decreasing", "new", noise, run_number=int(run_number))
    
def test():
    # train_single_model_on_moons("decreasing", "new", 0.05, run_number=int(np.array([0])))
    pass

if __name__ == "__main__":
    # main()
    main()