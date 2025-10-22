import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import pathlib as pl
import matplotlib.pyplot as plt
import pandas as pd

import os, sys


def train_model(model, train_data, test_data, savepath, SAVE_STATES = False, save_everyth_epoch = 100):
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    #TODO make this optimizer customable
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    for epoch in trange(EPOCHS):
        model.train()
        running_loss = 0
        num_correct = 0
        total_samples = 0
        for i, (x, y) in enumerate(train_data):
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
