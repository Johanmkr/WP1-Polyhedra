
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt


# Local imports
from src_experiment import moon_path, wbc_path, get_model, train_model, get_moons_data, get_wbc_data

saves_pr_run = 15

def train_model_on_moons(
    arch: str,
    dropout: float = 0.0,
    noise: float = 0.0,
    run_number: int = 0,
    epochs = 75,
    SAVE_STATES = False,
    RETURN_STATES = False,
):
    # Create savepath
    savepath = moon_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
    
    # Get data
    train_loader, test_loader = get_moons_data(feature_noise=noise, batch_size=32)


    # Train model
    run_results = train_model(
        model=get_model(f"{arch}_moon", dropout=dropout, seed=run_number),
        train_data=train_loader,
        test_data=test_loader,
        epochs=epochs,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=epochs//saves_pr_run,
    ) 
    return run_results
    
def train_model_on_wbc(
    arch: str,
    dropout: float = 0.0,
    noise: float = 0.0,
    run_number: int = 0,
    epochs = 75,
    SAVE_STATES = False,
    RETURN_STATES = False,
):
    # Create savepath
    savepath = wbc_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
    
    # Get data
    train_loader, test_loader = get_wbc_data(label_noise = noise, batch_size=32)
   
    # Train model
    run_results = train_model(
        model=get_model(f"{arch}_wbc", dropout=dropout, seed=run_number),
        train_data=train_loader,
        test_data=test_loader,
        epochs=epochs,
        savepath=savepath,
        SAVE_STATES=SAVE_STATES,
        RETURN_STATES=RETURN_STATES,
        save_everyth_epoch=epochs//saves_pr_run,
    ) 
    return run_results


def test(arch = "small",
    dropout = 0.0,
    noise = 0.0,
    run_number = 0):

    print(f"Testing moons and wbc with \narch={arch}\ndropout={dropout}\nnoise={noise}\nrun_number={run_number}")

    res1 = train_model_on_moons(
        arch=arch,
        dropout=dropout,
        noise=noise,
        run_number=run_number)
    res2 = train_model_on_wbc(
        arch=arch,
        dropout=dropout,
        noise=noise,
        run_number=run_number)

    fig, (ax1, ax2) = plt.subplots(2,1, sharex=True, figsize=(15,10))
    res1[0].plot(ax=ax1)
    res2[0].plot(ax=ax2)
    ax1.set_title("Moons dataset")
    ax2.set_title("WBC dataset")
    ax2.set_xlabel("Epoch")
    ax1.set_ylabel("Loss / Accuracy")
    ax2.set_ylabel("Loss / Accuracy")
    fig.suptitle(f"Training curves for moons (top) and wbc (bottom) \narch={arch}, dropout={dropout}, noise={noise}, run_number={run_number}")
    plt.show()
    
def inspect_run_numbers():
    for arch in ["small", "decreasing"]:
        for run_number in range(21):
            test(arch=arch, run_number=run_number)
            
            
def train_both_efficiently(epochs=75):
    dropouts = [0.0, 0.05, 0.1, 0.15, 0.2]
    noises = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    architectures = ["small", "decreasing"]
    run_numbers = [0,2,4,5,6,8,9,10,11,12,13,14,15,18,19] # 15 runs that converge well
    
    # Total runs: 5*7*2*15 = 1050 runs for both datasets
    # Loop through data configurations first to load them as few times as possible
    i = 1
    tot = len(dropouts)*len(noises)*len(architectures)*len(run_numbers)
    for noise in noises:
        moon_data = get_moons_data(feature_noise=noise, batch_size=32)
        wbc_data = get_wbc_data(label_noise=noise, batch_size=32)
        # Loop over model configs
        for arch in architectures:
            for dropout in dropouts:
                for run_number in run_numbers:
                    print(f"\nRun {i} of {tot}")
                    print(f"Training moons and wbc with \narch={arch}\ndropout={dropout}\nnoise={noise}\nrun_number={run_number}")
                    i += 1

                    # Moons
                    savepath_moon = moon_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
                    train_model(
                        model=get_model(f"{arch}_moon", dropout=dropout, seed=run_number),
                        train_data=moon_data[0],
                        test_data=moon_data[1],
                        epochs=epochs,
                        savepath=savepath_moon,
                        SAVE_STATES=True,
                        RETURN_STATES=False,
                        save_everyth_epoch=epochs//saves_pr_run,
                    ) 
                    
                    # WBC
                    savepath_wbc = wbc_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
                    train_model(
                        model=get_model(f"{arch}_wbc", dropout=dropout, seed=run_number),
                        train_data=wbc_data[0],
                        test_data=wbc_data[1],
                        epochs=epochs,
                        savepath=savepath_wbc,
                        SAVE_STATES=True,
                        RETURN_STATES=False,
                        save_everyth_epoch=epochs//saves_pr_run,
                    )

    
    
if __name__ == "__main__":
    # main()
    # test()
    # dropouts = [0.0, 0.05, 0.1, 0.15, 0.2]
    # noises = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    # test(arch="small", dropout=0.2, noise=0.4, run_number=11)
    train_both_efficiently(epochs=75)