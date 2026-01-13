# Import necessary libraries
# from numpy import number
from tqdm import trange
from src_experiment import moon_path, wbc_path, get_moons_data, get_wbc_data
import geobin as gb
import os
import torch
from pathlib import Path

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle
    
# Helper functions to read and wrtie pickle objects.

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def open_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj


def find_and_store_counts(basepath: Path,
                          data: torch.utils.data.DataLoader,
                          epochs: list[int],
                          overwrite=False):

    savepath = basepath
    number_count_path = savepath/"number_counts_per_epoch.pkl"
    
    def _run_and_save_counts():
        # trees = {} # Dict to store trees. Epoch number as keys
        

        ncounts_per_epoch = {}
        inference_data = data
        
        # Main loop:
        # Iterate through epochs
        #   1. Load the state dict
        #   2. Generate the tree object
        #   3. Save tree to 
        
        for epoch in epochs:
            #Load state dict
            state_dict_path = savepath / "state_dicts" / f"epoch{epoch}.pth"
            state_dict = torch.load(state_dict_path)
            # Initialize tree
            tree = gb.RegionTree(state_dict)
            # Build tree
            tree.build_tree(verbose=False)
            # trees[epoch] = tree
            
            
            # Pass data through tree
            tree.pass_dataloader_through_tree(inference_data)
            # Find number counts
            
            tree.collect_number_counts()
            ncounts = tree.get_number_counts()
            ncounts_per_epoch[epoch] = ncounts
                
                
        # Save objects
        
        # Un-comment to save trees
        # save_object(trees, savepath/"trees.pkl")

        # Save the number counts. 
        save_object(ncounts_per_epoch, number_count_path)
    
    # Check if number counts exists
    
    if os.path.exists(number_count_path):
        if overwrite:
            print("Old number counts will be overwritten")
            _run_and_save_counts()
        else:
            print("Number counts already exists.")
    else:
        print("Finding number counts...")
        _run_and_save_counts()


# def run_moon_number_counts(
#     model_name="small",
#     dataset_name="new",
#     noise_level=0.05,
#     run_number=0
# ):
#     basepath = get_storage_path("moons", model_name=model_name, dataset_name=dataset_name, noise_level=noise_level, run_number=run_number)
#     data = get_data("moons", "testing", noise=noise_level)
#     find_and_store_counts(basepath, data, epochs=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,124], overwrite=True)
    
    
# def run_wbc_number_counts(
#     model_name="small",
#     run_number=0
# ):
#     basepath = get_storage_path("wbc", model_name=model_name, run_number=run_number)
#     data = get_data("breast_cancer", "testing", batch_size=75)
#     find_and_store_counts(basepath, data, epochs=[0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400,420,440,460,480,499], overwrite=True)
    
# def run_all_moon_counts():
#     run_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
#     for noise in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
#         for number in run_numbers:
#             print(f"\nNoise: {noise}\nRun {number}/{run_numbers[-1]}")
#             run_moon_number_counts(
#                 model_name="decreasing",
#                 dataset_name="new",
#                 noise_level = noise,
#                 run_number = number,
#         )
   
# def run_all_wbc_counts():
#     run_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14]
#     for model_name in ["small", "decreasing"]:
#         for number in run_numbers:
#             print(f"\nModel: {model_name}\nRun {number}/{run_numbers[-1]}")
#             run_wbc_number_counts(
#                 model_name=model_name,
#                 run_number = number,
#         )  
         
# def test():
#     run_wbc_number_counts(
#         model_name="small",
#         run_number=0
#     )
    
    
def main():
    dropouts = [0.0, 0.05, 0.1, 0.15, 0.2]
    noises = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    architectures = ["small", "decreasing"]
    run_numbers = [0,2,4,5,6,8,9,10,11,12,13,14,15,18,19] # 15 runs that converge well
    epochs = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,74]
    
    # Total runs: 5*7*2*15 = 1050 runs for both datasets
    # Loop through data configurations first to load them as few times as possible
    i = 1
    tot = len(dropouts)*len(noises)*len(architectures)*len(run_numbers)
    for noise in noises:
        moon_data = get_moons_data(feature_noise=noise, batch_size=32)
        wbc_data = get_wbc_data(label_noise=noise, batch_size=32)
        moon_testing = moon_data[1] # Testing data
        wbc_testing = wbc_data[1] # Testing data
        # Loop over model configs
        for arch in architectures:
            for dropout in dropouts:
                for run_number in run_numbers:
                    print(f"\nRun {i} of {tot}")
                    print(f"Finding and storing number counts for:\narch={arch}\ndropout={dropout}\nnoise={noise}\nrun_number={run_number}")
                    i += 1
                    # Moons
                    path_moon = moon_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
                    find_and_store_counts(path_moon, moon_testing, epochs=epochs, overwrite=False)
                    # WBC
                    path_wbc = wbc_path(arch=arch, dropout=dropout, noise=noise, run_number=run_number)
                    find_and_store_counts(path_wbc, wbc_testing, epochs=epochs, overwrite=False)
    
if __name__=="__main__":
    main()