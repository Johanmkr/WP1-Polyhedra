from __future__ import annotations
import numpy as np
import itertools
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd
from scipy.optimize import linprog
from typing import Dict, List, Optional, Tuple, Iterable
from scipy.optimize import linprog
import sys
import os
import matplotlib.pyplot as plt
import time

sys.path.append('.')

from geobin_py import Region, Tree, RegionTree, TreeNode


from src_experiment import get_moons_data, train_model, NeuralNet
train, test = get_moons_data(feature_noise=.2)
all_features = []
all_labels = []

for features, labels in train:
    all_features.append(features.detach().cpu())
    all_labels.append(labels.detach().cpu())

# Combine batches into one large array
import torch
X = torch.cat(all_features, dim=0).numpy()
y = torch.cat(all_labels, dim=0).numpy()

plt.figure(figsize=(8,8))

# 3. Plot using the first two columns (dimensions) of the features
# X[:, 0] is Feature 1, X[:, 1] is Feature 2
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolor='k', s=50)

plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.grid(True)
plt.show()

from visualization import plot_training
from src_experiment import get_test_data
hidden_sizes = [0,9,7,5,3,1]

ActNet = NeuralNet(
    input_size = 2,
    hidden_sizes = hidden_sizes[1:-1],
    num_classes=1,
    seed=3,
)

res, = train_model(ActNet, train, test, epochs=41, save_everyth_epoch=10, SAVE_STATES=True, savepath=get_test_data())
plot_training(res)

trees = {}
tot_start = time.time()
for epoch in [0,10,20,30,40]:
    state_dict_path = get_test_data() / "state_dicts" / f"epoch{epoch}.pth"
    state = torch.load(state_dict_path)
    start = time.time()
    print(f"\n--- Epoch {epoch} ---")
    tree = Tree(state)
    tree.construct_tree(verbose=True)
    trees[epoch] = tree
    end = time.time()
    print(f"Duration: {end-start:.2f} s")
tot_end = time.time()
print(f"Total duration: {tot_end-tot_start:.2f} s")

import itertools
t0 = trees[0]
# ot0 = oldtrees[0]
n_theo_all = 2**np.cumsum(np.array(hidden_sizes))

# prev_layer_nodes = [ot0.root]
for i in range(t0.L):
    # old_nodes = []
    # old_nodes.extend(itertools.chain.from_iterable([n.get_children() for n in prev_layer_nodes]))
    n_new = len(t0.get_regions_at_layer(i+1))
    # n_old = len(old_nodes)
    n_theo = n_theo_all[i+1]
    
    print(f"\nLayer: {i+1}\nRegions (new): {n_new}\nRegions (tot): {n_theo}\n")
    # print(f"\nLayer: {i+1}\nRegions (new): {n_new}\nRegions (old): {n_old}\nRegions (tot): {n_theo}\n")
    # prev_layer_nodes = old_nodes
    
    
from visualization import plot_epoch_layer_grid
plot_epoch_layer_grid(trees, bound=2.1)