import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import pathlib as pl
import matplotlib.pyplot as plt
import pandas as pd

def train_model(model, train_data, test_data, epochs, savepath, experiment_name, SAVE_STATES = False, save_everyth_epoch = 10):
    train_loss = np.zeros((epochs))
    train_accuracy = np.zeros((epochs))
    test_loss = np.zeros((epochs))
    test_accuracy = np.zeros((epochs))

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in trange(epochs):
        model.train()
        running_loss = 0 
        num_correct = 0 
        total_samples = 0 

        for i, (x,y) in enumerate(train_data):
            optimizer.zero_grad()
            x = x.float()
            y = y.unsqueeze(1).float()
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_correct += torch.eq(y_hat.round().bool(), y.bool()).sum()
            total_samples += y.size(0)

        train_loss[epoch] = running_loss/len(train_data)
        train_accuracy[epoch] = num_correct/total_samples

       
        model.eval()
        with torch.no_grad():
            running_loss = 0 
            num_correct = 0 
            num_samples = 0 

            for i, (x,y) in enumerate(test_data):
                x = x.float()
                y = y.unsqueeze(1).float()
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                
                running_loss += loss.item()
                num_correct += torch.eq(y_hat.round().bool(),y.bool()).sum()
                total_samples += y.size(0)
            
            test_loss[epoch] = running_loss/len(test_data)
            test_accuracy[epoch] = num_correct/num_samples

        if SAVE_STATES and (epoch % save_everyth_epoch == 0 or epoch == epochs-1):
            torch.save(model.state_dict(), savepath / f"{experiment_name}_epoch{epoch}.pth")
        
    experiment_results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    if SAVE_STATES:
        frame = pd.DataFrame.from_dict(experiment_results)
        frame.to_pickle(savepath/".."/f"{experiment_name}_training_data.pkl")
    return experiment_results

if __name__=="__main__":
   pass
