
import numpy as np 
from pyparsing import Optional
from sklearn import tree
from tqdm import tqdm
import pathlib as pl
import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
try:
    import cpickle as pickle
except ModuleNotFoundError:
    import pickle
import torch
import os, sys
from .region_tree import RegionTree
from .tree_node import TreeNode
import pandas as pd
from collections import defaultdict
from src_experiment import get_path_to_moon_experiment_storage

def layer_wise_MI_from_number_counts(counts:pd.DataFrame):
    # See if dataframe has a column "total"
    if "total" not in counts.columns:
        classes = counts.columns.values[2:] # Two first columns are layer and region indices, the rest are class labels
        # Convert nans to zeros
        for cl in classes:
            counts[cl] = np.nan_to_num(counts[cl], nan=0.0)
        
        # Create new column with total number counts
        counts["total"] = counts.apply(lambda row: np.sum([row[cl] for cl in classes]), axis=1)
    else:
        classes = counts.columns.values[2:-1]
        
        
    # Separate out first layer data to find the classwise split of data
    first_layer_counts = counts[counts.layer_idx==1]
    data_per_class = {classes[i]: np.sum(first_layer_counts[classes[i]]) for i in range(len(classes))}
    
    # Sparse out counts just in case
    counts = counts[counts["total"]!=0]
    
    # Find total number of points
    N = sum(data_per_class.values())
    
    # Total samples per region
    n_w = counts["total"]
    
    # Create column for region-wise-mutual-information
    counts["rwmi"] = 0
    
    # Perform class-wise addition of MI-term
    for cl in classes:
        n_kw = counts[cl] # Samples of class cl in each region
        n_k = data_per_class[cl] # Total samples of class cl
        
        # Add MI-term
        counts["rwmi"] += np.where(n_kw > 0, n_kw / N * np.log((N*n_kw)/(n_k*n_w)), 0) # Add zero if n_kw = 0 to avoid log(0). 
        
    # Add up every region in each layer to find the layer-wise MI
    lwmi = {}
    for layer in set(counts["layer_idx"]):
        lwmi[layer] = sum(counts[counts["layer_idx"]==layer]["rwmi"])
        
    return lwmi



        
        
class EstimateQuantities1Run:
    def __init__(self, model_name = "small_uniform",
    dataset_name = "small",
    noise_level = 0.0,
    run_number = 1):
        self.model_name = model_name
        self.noise_level = noise_level
        self.run_number = run_number
        
        # # New variables
        # self.num_layers = None
        # self.classes = None
        
        # Step 1 - Load the ncounts dictionary and create list of epochs
        
        # Step 2 - For each epoch do:
        #   a) Sort values by layer idx
        #   b) Run consistency check
        #   c) Estimate quantites per class per epoch
        #   d) Convert this estimate to layer-wise estimates. 
        
        # Step 3 - Make a layer-wise dictionary with the quantites per epoch, ready for plotting. 
        
        # Step 4 - Something else?
        
        
        # 1:
        # Path to data
        self.data_path = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
        
        # Get number counts
        self.ncounts = self._open_object(self.data_path) # Dict with epochs as keys
        
        # Get list of epoch
        self.epochs = np.array(list(self.ncounts.keys()))
        
        estimates = {}
        
        # 2 - Iterate over all epochs
        for epoch, frame in self.ncounts.items():
            # Sort frame by layer idx
            frame = frame.sort_values(by=["layer_idx", "region_idx"]).reset_index(drop=True)
            
            # Run consistency
            frame, num_layers, classes, data_per_class = self._run_consistency_check_on_single_frame(frame)
            
            # Estimate quantites
            estimates_per_layer = pd.DataFrame()
            frame["MI_KL"] = 0.0
            N = sum(data_per_class.values()) # Total number of points
            n_w = np.array(frame["total"]) # Total number of points pr. region
            for cl in classes: 
                n_kw = np.array(frame[cl]) # Number of points pr. class pr. region
                n_k = np.array(data_per_class[cl]) # Total number of points pr. class
                
                #### MI KL ####
                # Two-step log to avoid 0 as argument
                logterm = N*n_kw / (n_k*n_w)
                logterm = np.log(logterm, where=logterm>0)
                frame["MI_KL"] += n_kw / N * logterm
            
            for layer in range(1, num_layers+1):
                estimates_per_layer["layer"] = layer
                estimates_per_layer["MI_KL"] = sum(frame[frame.layer_idx==layer]["MI_KL"])
                
            
            
        
        
    def _run_consistency_check_on_single_frame(self, frame):
        # Check if last column is "total" and check if the sum of total counts is the same for each layer.
        assert frame.columns[ -1 ] == "total", "Last column should be 'total'"

        # Find the class columns as the third to the second last columns
        class_columns = frame.columns[ 2 : -1 ]
        classes = class_columns.tolist()

        # Check that the horisontal sum of classes equals the total counts for each row
        # Find sum of all classes in one single row (region)
        row_sums = frame[classes].sum(axis=1)
        totals = frame["total"]
        # Check if all row sums equal total counts
        assert all(row_sums == totals), "Row sums of class counts should equal total counts"

        # Find number of unique layers
        num_layers = frame["layer_idx"].nunique()
        # Get total counts per layer
        total_counts_per_layer = frame.groupby("layer_idx")["total"].sum().values
        # Check if all total counts are the same
        assert all(total_counts_per_layer == total_counts_per_layer[0]), "Total counts should be the same for each layer"
        
        # Find data per class
        layer_counts = frame[frame.layer_idx == 0]
        data_per_class = {cl: np.sum(layer_counts[cl]) for cl in classes}

        return frame, num_layers, classes, data_per_class
    
    
    # Method to open and read pickled ojects. 
    def _open_object(self, filename):
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        return obj

if __name__ == "__main__":
    pass
    
    
