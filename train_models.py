
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import torch

# Local imports
from src_experiment import get_new_path, get_new_data, train_model_multiclass, NeuralNet

# Experiment params
datasets = ["moons", "wbc", "wine", "hd", "car"]
noises = [0.0, 0.2, 0.4]
dropouts = [0.0, 0.1, 0.3, 0.5]
run_nrs = [1,2,3,4,5]

save_for_epochs = [0,2,4,6,8,10,20,30,40,50]
train_for_epochs = 51
model_hidden_sizes = [5,5,5]



def train_all():
    i = 1
    tot = len(datasets)*len(noises)*len(dropouts)*len(run_nrs)
    
    for dataset in datasets:
        for noise in noises:
            train_data, test_data = get_new_data(dataset, noise)
            
            # Get shape of data
            x_batch, y_batch = next(iter(train_data))
            input_size = x_batch.shape[1]

            all_train_labels = torch.cat([y for _, y in train_data])
            num_classes = len(torch.unique(all_train_labels))
            
            for dropout in dropouts:
                for run_nr in run_nrs:
                    # prints
                    print(f"\nRun {i}/{tot}")
                    print(f"Training ...\nDataset: {dataset}\nNoise: {noise}\nDropout: {dropout}\nRun nr: {run_nr}\n")
                    i += 1
                    
                    # Get savepath
                    savepath = get_new_path(dataset, noise, dropout, run_nr)
                    
                    # Initialise new model
                    model = NeuralNet(
                        input_size = input_size,
                        hidden_sizes = model_hidden_sizes,
                        num_classes = num_classes,
                        dropout = dropout,
                        seed = run_nr,
                    )
                    
                    # Train model
                    train_model_multiclass(
                        model = model,
                        train_data = train_data,
                        test_data = test_data,
                        epochs = train_for_epochs,
                        num_classes = num_classes,
                        savepath = savepath,
                        SAVE_STATES = True,
                        save_for_epochs = save_for_epochs,
                    )
                    # raise ValueError
                    

if __name__=="__main__":
    train_all()