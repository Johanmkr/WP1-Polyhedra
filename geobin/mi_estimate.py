import numpy as np 
from tqdm import tqdm
import pathlib as pl
import torch
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import pickle

import os, sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add parent directory to sys.path
sys.path.append(parent_dir)
import src.treenode as tn
ground_path = pl.Path(parent_dir) / "state_dicts"


params = {'legend.fontsize': 'x-large',
          'figure.figsize': (7, 4),
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

def estimate_mutual_information_from_number_counts(number_counts:dict, N:int, n0Y:int, n1Y:int):
    MI_layer = {}
    for layer, layer_counts in number_counts.items():
        MI = 0
        for region_counts in layer_counts:
            nk0 = region_counts[0] # Class 0 points in region
            nk1 = region_counts[1] # Class 1 points in region
            nkX = nk0 + nk1 # Sum. Total number of points in region.

            term1 = (nk0/N) * np.log(N*nk0/(nkX*n0Y)) if nk0 > 1e-5 else 0
            term2 = (nk1/N) * np.log(N*nk1/(nkX*n1Y)) if nk1 > 1e-5 else 0
            
            MI += (term1+term2)
        MI_layer[layer] = MI
    # print(MI_layer)
    return MI_layer


# Function to calculate the mutual information for each layer
def calc_MI(local_state_dict, class0, class1):
    # Create and build tree from state dict
    tree = tn.RegionTree(local_state_dict)
    tree.build_tree()
    
    # Run class 0 data through and store number counts
    tree.reset_counters()
    for input in tqdm(np.array(class0)):
        tree.pass_input_through_tree(input)
    tree.store_counters(reset=True)
    
    # Run class 1 data through ad store number counts
    for input in tqdm(np.array(class1)):
        tree.pass_input_through_tree(input)
    tree.store_counters(reset=True)
    
    # Extract number counts of non-zero nodes
    number_counts = tree.get_number_counts()
    print(number_counts)
    
    # Estimate mutual information:
    n0Y = len(class0)
    n1Y = len(class1)
    N = n0Y + n1Y
    local_MI = estimate_mutual_information_from_number_counts(number_counts, N, n0Y, n1Y)
    print(local_MI)
    return local_MI


def estimate_MI_from_experiment(exp_name, X, y, store=True):
    # Find state dicts
    state_dicts = {}
    for filename in os.listdir(ground_path/exp_name):
        if filename.endswith(".pth"):
            model_name = filename[:-4]  # Remove .pth extension
            if model_name[:6] == "mm_ssd":
                epoch = model_name[12:]
            else:
                epoch = model_name[11:]  # Extract epoch number
            # from IPython import embed; embed()
            # print(f"Loading model from {model_name}")
            state_dicts[epoch] = torch.load(ground_path/exp_name/filename)
    
    # Define the data classes
    class0 = X[y==0]
    class1 = X[y==1]
    
    # for each state dict, calculate MI for all layers
    MI_pr_epoch = {}
    
    for epoch, state_dict in state_dicts.items():
        MI_pr_epoch[epoch] = calc_MI(state_dict, class0, class1)
        
    # Get ordered list of epochs as int
    epochs = state_dicts.keys()
    epochs = [eval(epoch) for epoch in epochs]
    epochs = sorted(epochs)
    
    # Convert data to layer wise format to make plotting easier
    layer_wise_data = {}
    # epochs = []
    for epoch in epochs: 
        layer_data = MI_pr_epoch[f"{epoch}"]
        for layer, data in layer_data.items():
        # if epoch in [0, "0"]:
        #     layer_wise_data[layer] = [data]
        # else:
        #     layer_wise_data[layer].append(data)
            try:
                layer_wise_data[layer].append(data)
            except KeyError:
                layer_wise_data[layer] = [data]
        # epochs.append(eval(epoch))
    store_data = { 
            "exp_name": exp_name,
            "epochs": epochs,
            "layer_wise_data": layer_wise_data
            }
    if store:
        storepath = ground_path/exp_name/"mi_information.pkl"
        with open(storepath, "wb") as fp:
            pickle.dump(store_data, fp)
            print("MI-estimation saved successfully to file")
    return store_data


titles={
    "sm_sd": "MI for small model, smooth data", 
    "lm_sd": "MI for large model, smooth data",
    "lm_nd": "MI for large model, noisy data",
    "sm_nd": "MI for small model, noisy data",
    "mm_sd": "MI for smooth data",
    "mm_ssd": "MI for semi-smooth data",
    "mm_nd": "MI for noisy data"
}

def plot_layer_wise_data(layer_wise_dict, exp_name):
    
    fig, ax = plt.subplots()
    markers = ["o", "v", "*", ">"]
    for i, (layer, data) in enumerate(layer_wise_dict["layer_wise_data"].items()):
        if layer in [0, "0"] or layer in [5, "5"]:
            continue
        ax.plot(layer_wise_dict["epochs"][::4], data[::4], ls="-", marker=markers[i-1], lw=1.5, markersize=10, label=f"Layer: {layer}")
    # ax.set_title(titles[exp_name])
    ax.set_xlabel("Epoch")
    ax.set_ylabel(r"$MI(L,Y)$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(pl.Path(parent_dir)/"figures"/f"MI_{layer_wise_dict['exp_name']}.pdf")
    plt.show()

if __name__ == "__main__":
    X_val, y_val = make_moons(n_samples=5000, noise=0.05, random_state=111)
    X_val_noise, y_val_noise = make_moons(n_samples=5000, noise=2, random_state=111)
    
    # lwd1 = estimate_MI_from_experiment("sm_sd", X_val, y_val)
    # # plot_layer_wise_data(lwd1)
    # lwd2 = estimate_MI_from_experiment("lm_sd", X_val, y_val)
    # # plot_layer_wise_data(lwd2)
    # lwd3 = estimate_MI_from_experiment("sm_nd", X_val_noise, y_val_noise)
    # # plot_layer_wise_data(lwd3)
    # lwd4 = estimate_MI_from_experiment("lm_nd", X_val_noise, y_val_noise)
    # # plot_layer_wise_data(lwd4)
    
    # X5, y5 = make_moons(n_samples=10000, noise=0.05, random_state=111)
    # lwd5 = estimate_MI_from_experiment("mm_sd", X5, y5)
    # plot_layer_wise_data(lwd5)
    
    # X6, y6 = make_moons(n_samples=10000, noise=0.25, random_state=111)
    # lwd6 = estimate_MI_from_experiment("mm_ssd", X6, y6)
    # plot_layer_wise_data(lwd6)
    
    # X7, y7 = make_moons(n_samples=10000, noise=0.5, random_state=111)
    # lwd7 = estimate_MI_from_experiment("mm_nd", X7, y7)
    # plot_layer_wise_data(lwd7)
    
    
    for exp_name in ["mm_sd", "mm_ssd", "mm_nd"]:
        with open(ground_path/exp_name/"mi_information.pkl", "rb") as fp:
            lwd = pickle.load(fp)
        plot_layer_wise_data(lwd, exp_name)
    
    
    
    
