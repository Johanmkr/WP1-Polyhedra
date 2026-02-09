# Import necessary libraries
# from numpy import number
from tqdm import trange
import os
import torch
from pathlib import Path
import pandas as pd
import time

try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle
    
from src_experiment import get_new_path, get_new_data
import geobin_py as gb
    
# Helper functions to read and wrtie pickle objects.

def save_object(obj, filename):
    with open(filename, 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def open_object(filename):
    with open(filename, 'rb') as inp:
        obj = pickle.load(inp)
    return obj

def save_object_csv(obj, filename):
    # index=False is usually preferred unless the index contains important data
    df = pd.DataFrame.from_dict(obj)
    df.to_csv(filename, index=False)
        
def open_object_csv(filename):
    return pd.read_csv(filename)

def find_and_store_counts(basepath: Path,
                          data: torch.utils.data.DataLoader,
                          epochs: list[int],
                          overwrite=False):

    savepath = basepath
    
    number_count_path = savepath/"number_counts_per_epoch.pkl"
    # number_count_path = savepath/"number_counts_per_epoch.csv"
    
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
            print(f"Building for epoch: {epoch} in {epochs}")
            #Load state dict
            state_dict_path = savepath / "state_dicts" / f"epoch{epoch}.pth"
            state_dict = torch.load(state_dict_path)
            # Initialize tree
            
            start = time.time()
            tree = gb.RegionTree(state_dict)
            # Build tree
            tree.build_tree(verbose=True)
            # trees[epoch] = tree
            
            
            # Pass data through tree
            tree.pass_dataloader_through_tree(inference_data)
            # Find number counts
            
            tree.collect_number_counts()
            ncounts = tree.get_number_counts()
            ncounts_per_epoch[epoch] = ncounts
            end = time.time()
            dur = end - start
            print("----------------------------------------------------")
            print(f"Duration of ncounts for epoch {epoch}: {dur:.3f} s")
            print("----------------------------------------------------")
                    
                
        # Save objects
        
        # Un-comment to save trees
        # save_object(trees, savepath/"trees.pkl")

        # Save the number counts.
        
        # PKL saving
        save_object(ncounts_per_epoch, number_count_path)
        
        # CSV saving
        # save_object_csv(ncounts_per_epoch, number_count_path)
    
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
    
def main():
    # Experiment params
    datasets = ["moons", "wbc", "wine", "hd", "car"]
    noises = [0.0, 0.2, 0.4]
    dropouts = [0.0, 0.1, 0.3, 0.5]
    run_nrs = [1,2,3,4,5]

    save_for_epochs = [0,2,4,6,8,10,20,30,40,50]
    i = 1
    tot = len(datasets)*len(noises)*len(dropouts)*len(run_nrs)

    for dataset in datasets:
        for noise in noises:
            train_data, test_data = get_new_data(dataset, noise)
            
            for dropout in dropouts:
                for run_nr in run_nrs:
                    # prints
                    print(f"\nRun {i}/{tot}")
                    print(f"Finding and storing number counts for\nDataset: {dataset}\nNoise: {noise}\nDropout: {dropout}\nRun nr: {run_nr}\n")
                    i += 1
                    
                    # Get savepath
                    savepath = get_new_path(dataset, noise, dropout, run_nr)
                    find_and_store_counts(
                        basepath = savepath,
                        data = test_data,
                        epochs = save_for_epochs,
                        overwrite=False,
                    )
                    # raise ValueError
                    
    
    
    
if __name__=="__main__":
    main()