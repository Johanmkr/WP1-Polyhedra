
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Local imports
from src_experiment import get_storage_path, get_model, train_model, get_data


def train_model_on_moons(
    model_name: str,
    dataset_name: str,
    noise_level: float,
    run_number: int = 0,
    epochs = 125,
    SAVE_STATES = False,
    RETURN_STATES = False,
    save_everyth_epoch = 5,
):
    # Create savepath
    savepath = get_storage_path("moons", model_name=model_name, dataset_name=dataset_name, noise_level=noise_level, run_number=run_number)
    
    # Get data
    train_loader = get_data("moons", "training", noise=noise_level)
    test_loader = get_data("moons", "testing", noise=noise_level)
   
    
    # Train model
    run_results = train_model(
        model=get_model(f"{model_name}_moon", seed=run_number),
        train_data=train_loader,
        test_data=test_loader,
        epochs=epochs,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=save_everyth_epoch,
    ) 
    return run_results
    
def train_model_on_wbc(
    model_name: str,
    run_number: int = 0,
    epochs = 500,
    SAVE_STATES = False,
    RETURN_STATES = False,
    save_everyth_epoch = 20,
):
    # Create savepath
    savepath = get_storage_path("wbc", model_name=model_name, run_number=run_number)
    
    # Get data
    train_loader = get_data("breast_cancer", "training", batch_size=75)
    test_loader = get_data("breast_cancer", "testing", batch_size=75)
   
    
    # Train model
    run_results = train_model(
        model=get_model(f"{model_name}_wbc", seed=run_number),
        train_data=train_loader,
        test_data=test_loader,
        epochs=epochs,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=save_everyth_epoch,
    ) 
    return run_results
    
    
def train_moons():
    for noise in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for run_number in np.arange(35):
            print(f"Noise: {noise}\nRun {run_number}/{np.arange(35).max()}")
            train_model_on_moons("decreasing", "new", noise, run_number=int(run_number))
    
def train_wbc():
    for model_name in ["small", "decreasing"]:
        for run_number in np.arange(15):
            print(f"Run {run_number}/{np.arange(10).max()}")
            train_model_on_wbc(model_name, run_number=int(run_number), SAVE_STATES=True)

def test():
    res = train_model_on_wbc("small", SAVE_STATES=True)
    plt.figure()
    res[0].plot()
    plt.show()

if __name__ == "__main__":
    # main()
    train_wbc()