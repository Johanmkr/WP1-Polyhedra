
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

     
        
class EstimateQuantities1Run:
    def __init__(self, model_name = "small_uniform",
    dataset_name = "small",
    noise_level = 0.0,
    run_number = 1,
    calculate=False):
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
        self.data_dir = get_path_to_moon_experiment_storage(model_name, dataset_name, noise_level, run_number)
        self.data_path = self.data_dir / "number_counts_per_epoch.pkl"
        
        # Get number counts
        self.ncounts = self._open_object(self.data_path) # Dict with epochs as keys
        
        # Get list of epoch
        self.epochs = np.array(list(self.ncounts.keys()))
        
        self.estimates = {
            "MI_KL": [],
            "MI_IS": [],
        }
        
        if calculate:
            self.calculate_estimates()
        
        
    def get_estimates(self):
        return self.estimates
        
        
    def calculate_estimates(self):
        # 2 - Iterate over all epochs
        for epoch, frame in self.ncounts.items():
            
            # Sort frame by layer idx
            frame = frame.sort_values(by=["layer_idx", "region_idx"]).reset_index(drop=True)
            
            # Run consistency
            frame, num_layers, classes, n_k = self._run_consistency_check_on_single_frame(frame)
            
            # Estimate quantites
            # estimates_per_layer = pd.DataFrame()
            
            # N = sum(data_per_class.values()) # Total number of points
            # n_w = np.array(frame["total"]) # Total number of points pr. region
            
            
            # Find N, n_w, n_kw, n_k, m_w, m_kw, m_k
            n_w = np.array(frame['total']) # Points pr. region
            n_kw = np.array(frame[classes]) # Points pr. class
            N = n_k.sum() # Total number of points
            
            # Make probability masses
            m_w = np.expand_dims(n_w / N, axis=1)
            m_kw = n_kw / N
            m_k = n_k / N

            # Estimate Quantites
            for estimate, results in self.estimates.items():
                # Calculate estimate
                frame[estimate] = self._individual_estimates(estimate, m_w, m_kw, m_k, N)
                result = (
                    frame.groupby('layer_idx')[estimate].sum()
                    .rename(lambda i: f"l{i}")
                    .to_frame()
                    .T
                )
                result.insert(0, "epoch", epoch)
                result.index.name = None
                results.append(result)
                
        # 3 - Make a layer-wise dictionary with the quantites per epoch, ready for plotting. 
        for estimate, results in self.estimates.items():
            newframe = pd.concat(results, ignore_index=True)
            newframe = newframe.rename_axis(None, axis=1)
            self.estimates[estimate] = newframe 
            
    
    def _individual_estimates(self, estimate, m_w, m_kw, m_k, N):
        match estimate:
            case "MI_KL": # Kullback-Leibler diverence
                logterm = m_kw / (m_w @ m_k)
                logterm = np.log(logterm, where=logterm > 0)
                return (m_kw * logterm).sum(axis=1)
            case "MI_IS": # Itakura-Saito divergence
                return 0 #TODO
            case _:
                raise ValueError("Estimate identifier not found!")
        
        
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
        layer_counts = frame.loc[frame["layer_idx"] == 1, classes]
        n_k1 = layer_counts.sum(axis=0).to_numpy()[None, :]
        for layer in range(1, num_layers+1):
            layer_counts = frame.loc[frame["layer_idx"] == layer, classes]
            n_k = layer_counts.sum(axis=0).to_numpy()[None, :]
            
            # Check that class division is the same for all layers
            assert np.all(n_k1 == n_k)
            


        return frame, num_layers, classes, n_k
    
    
    # Method to open and read pickled ojects. 
    def _open_object(self, filename):
        with open(filename, 'rb') as inp:
            obj = pickle.load(inp)
        return obj

if __name__ == "__main__":
    pass
    
    
