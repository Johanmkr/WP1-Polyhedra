# Import necessary libraries
# from numpy import number
from tqdm import trange
from src_experiment import get_args, createfolders, moons_models, train_model, datasets, get_path_to_moon_experiment_storage, get_specific_moon_state_dict
import geobin as gb

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




def find_and_store_trees(model_name: str,
                         dataset_name: str,
                         noise_level: float,
                         run_number: int,
                         epochs: list[int],
                         count_numbers=False):
    savepath = get_path_to_moon_experiment_storage(model_name=model_name,dataset_name=dataset_name, noise_level=noise_level, run_number=run_number)
    
    trees = {} # Dict to store trees. Epoch number as keys
    
    if count_numbers:
        ncounts_per_epoch = {}
        inference_data = datasets[dataset_name][noise_level]["inference"]
    
    # Main loop:
    # Iterate through epochs
    #   1. Load the state dict
    #   2. Generate the tree object
    #   3. Save tree to 
    
    for epoch in epochs:
        #Load state dict
        state_dict = get_specific_moon_state_dict(model_name=model_name, dataset_name=dataset_name, noise_level=noise_level, run_number=run_number, epoch=epoch)
        # Initialize tree
        tree = gb.RegionTree(state_dict)
        # Build tree
        tree.build_tree(verbose=True)
        trees[epoch] = tree
        
        if count_numbers:
            # Pass data through tree
            tree.pass_dataloader_through_tree(inference_data)
            # Find number counts
            tree.collect_number_counts()
            ncounts = tree.get_number_counts()
            ncounts_per_epoch[epoch] = ncounts
            
            
    # Save objects
    save_object(trees, savepath/"trees.pkl")
    if count_numbers:
        save_object(ncounts_per_epoch, savepath/"number_counts_per_epoch.pkl")
        
        
def main():
    epochs=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,249]
    model_names = moons_models.keys()
    dataset_names = datasets.keys()
    noises = datasets[[key for key in dataset_names][0]].keys()
    
    run_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    
    remaining_dataset_names = ["decreasing", "increasing"]
    # Main loop
    for model in model_names:
        for dataset in dataset_names:
            for noise in noises:
                print(f"\nModel: {model}\nDataset: {dataset}\nNoise level: {noise}")
                for number in run_numbers:
                    print(f"Run number {number}/{run_numbers[-1]}")
                    find_and_store_trees(
                    model_name=model,
                    dataset_name=dataset,
                    noise_level = noise,
                    run_number=number,
                    epochs=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,249],
                    count_numbers=True
                    )

    
    
def test():
    run_numbers = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
    for number in run_numbers:
        find_and_store_trees(
            model_name="decreasing",
            dataset_name="small",
            noise_level = 0.0,
            run_number = number,
            epochs=[0,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,249],
            count_numbers=True
    )
    
    
    
if __name__=="__main__":
    test()