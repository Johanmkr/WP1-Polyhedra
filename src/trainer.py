import torch.nn as nn
import torch
from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
from tqdm import trange

class MoonTrainer:
    def __init__(self, model):
        self.model = model
        
        # Generate data
        x, y = make_moons(1000, noise=0.1)
        self.x, self.y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
        
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters())
        
        # Store training data
        self.losses = []
        self.hidden_layers = []
        self.output = []
        
        self.trained_epochs = 0
        
    def train(self, epochs=1000):
        for epoch in trange(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            
            out, hidden_layers = self.model(self.x)
            loss = self.criterion(out, self.y)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            # Append relevant data
            self.losses.append(loss.item())
            self.hidden_layers.append([h.cpu().detach().numpy() for h in hidden_layers])
            self.output.append(out.cpu().detach().numpy())
            self.trained_epochs += 1
            # if epoch % 100 == 0:
            #     print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
        print(f'Training complete after {self.trained_epochs} epochs.\nTotal training: {self.trained_epochs} epochs.')
    
        