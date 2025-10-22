import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import pathlib as pl
import matplotlib.pyplot as plt
import pandas as pd

import os, sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add parent directory to sys.path
sys.path.append(parent_dir)
from src import treenode as tn
from src import utils

from sklearn.datasets import make_moons


EPOCHS = 10000
criterion = nn.BCEWithLogitsLoss()

ground_path = pl.Path(parent_dir) / "state_dicts"

def train_model(experiment, SAVE_STATES = False, save_everyth_epoch = 100):
    exp_name = experiment["exp_name"]
    model = experiment["model"]
    data = experiment["data"]
    train_data = torch.tensor(data[0][0], dtype=torch.float32)
    train_target = torch.tensor(data[0][1], dtype=torch.float32)
    test_data = torch.tensor(data[1][0], dtype=torch.float32)
    test_target = torch.tensor(data[1][1], dtype=torch.float32)
    
    # Check savepath
    savedir = ground_path / exp_name
    if not savedir.is_dir():
        os.makedirs(savedir)
    
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    for epoch in trange(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data)
        loss = criterion(out, train_target.unsqueeze(1))
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        preds = torch.sigmoid(out) > 0.5
        accuracy = (preds.squeeze() == train_target).float().mean().item()
        train_accuracy.append(accuracy)
        
        model.eval()
        with torch.no_grad():
            out = model(test_data)
            loss = criterion(out, test_target.unsqueeze(1))
            preds = torch.sigmoid(out) > 0.5
            accuracy = (preds.squeeze() == test_target).float().mean().item()
            test_loss.append(loss.item())
            test_accuracy.append(accuracy)
        if SAVE_STATES and (epoch % save_everyth_epoch == 0 or epoch == EPOCHS-1):
            torch.save(model.state_dict(), savedir / f"{exp_name}_epoch{epoch}.pth")
        
    experiment_results = {
        "exp_name": exp_name,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    if SAVE_STATES:
        frame = pd.DataFrame.from_dict(experiment_results)
        frame.to_pickle(ground_path/exp_name/"training_data.pkl")
    return experiment_results

if __name__=="__main__":
   pass
