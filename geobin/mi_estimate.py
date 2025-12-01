
import numpy as np 
from tqdm import tqdm
import pathlib as pl
import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pickle
import torch
import os, sys
from .region_tree import RegionTree
from .tree_node import TreeNode
import pandas as pd

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
        
        
class MIQuantityEstimator:
    def __init__(self):
        pass

if __name__ == "__main__":
    pass
    
    
