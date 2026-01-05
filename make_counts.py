# Import necessary libraries
# from numpy import number
from tqdm import trange
from src_experiment import get_storage_path, get_data
import geobin as gb
import os
import torch
from pathlib import Path

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle


""" What should this script do?
    For one single run, the script should read the state dicts for every epoch and compute a trees object of all the corresponding trees. This object should be stored. 
    
    Further, a number count of each tree should be performed, but this require some additional data. 
    
    The data shohuld be created with the same noise level as the training and testing data, but different random seed. SHould import this from datasets. 

"""


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


def run_moon_number_counts(
    model_name="small",
    dataset_name="new",
    noise_level=0.05,
    run_number=0
):
    basepath = get_storage_path("moons", model_name=model_name, dataset_name=dataset_name, noise_level=noise_level, run_number=run_number)
    data = get_data("moons", "testing", noise=noise_level)
    find_and_store_counts(basepath, data, epochs=[0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100,105,110,115,120,124], overwrite=True)
    
def test():
    run_numbers = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34]
    for noise in [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        for number in run_numbers:
            print(f"\nNoise: {noise}\nRun {number}/{run_numbers[-1]}")
            run_moon_number_counts(
                model_name="decreasing",
                dataset_name="new",
                noise_level = noise,
                run_number = number,
        )
            
    
    
    
if __name__=="__main__":
    for noise in [0.05]:
        for number in [0]:
            # print(f"\nNoise: {noise}\nRun {number}/{run_numbers[-1]}")
            run_moon_number_counts(
                model_name="decreasing",
                dataset_name="new",
                noise_level = noise,
                run_number = number,
        )