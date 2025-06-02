# Based on Kristoffer Wickstrøm's notebook 'two_dimensional_representation_example.ipynb'

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons

x, y = make_moons(1000, noise=0.1)
x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

def to_np(x):
    return x.cpu().detach().numpy()
input = [x.numpy()]
target = [y.numpy()]
# Modfied model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()

        self.activation = nn.ReLU()

        self.l1 = nn.Linear(2, 3)
        self.l2 = nn.Linear(3, 2)
        self.out = nn.Linear(2, 1)

    def forward(self, x):

        layer1 = self.activation(self.l1(x))
        layer2 = self.activation(self.l2(layer1))
        out = torch.sigmoid(self.out(layer2).squeeze())

        return out, layer2, layer1
    
    
def train(model, EPOCHS = 10000):
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters())
    losses = []
    layer1 = []
    layer2 = []
    output = []
    
    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        out, l2, l1 = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        optimizer.zero_grad()
        
        # Append relevant data
        losses.append(to_np(loss))
        layer1.append(to_np(l1))
        layer2.append(to_np(l2))
        output.append(to_np(out))
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss.item()}')
    torch.save(model.state_dict(), 'model.pth')
    np.save('losses.npy', losses)
    np.save('input.npy', input)
    np.save('target.npy', target)
    np.save('layer1.npy', layer1)
    np.save('layer2.npy', layer2)
    np.save('output.npy', output)
    
def load_model():
    model = MLP()
    try:
        model.load_state_dict(torch.load('model.pth'))
    except FileNotFoundError:
        print("Model file not found. Training model.")
        train(model)
        model.load_state_dict(torch.load('model.pth'))
    return model


if __name__ == "__main__":
    model = load_model()
    # train(model)
    
    # Load losses and layers
    losses = np.load('losses.npy')
    input = np.load('input.npy')
    target = np.load('target.npy')
    layer1 = np.load('layer1.npy')
    layer2 = np.load('layer2.npy')
    output = np.load('output.npy')

    # from IPython import embed; embed()  
    # Plotting the loss
    plt.plot(losses)
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    