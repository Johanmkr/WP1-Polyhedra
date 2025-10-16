"""
Script that contains the MLP models used.
"""

import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_moons

# x, y = make_moons(1000, noise=0.1)
# x, y = torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# def to_np(x):
#     return x.cpu().detach().numpy()
# input = [x.numpy()]
# target = [y.numpy()]


# Modfied model
class MLP21(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super(MLP21, self).__init__()
        self.name = "MLP21"
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        self.h1 = nn.Linear(n_in, 2)
        self.out = nn.Linear(2, n_out)

    def forward(self, x):
        h1 = self.activation(self.h1(x))
        out = self.output_activation(self.out(h1).squeeze())

        return out, [h1]


class MLP31(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super(MLP31, self).__init__()
        self.name = "MLP31"
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        self.h1 = nn.Linear(n_in, 3)
        self.out = nn.Linear(3, n_out)

    def forward(self, x):
        h1 = self.activation(self.h1(x))
        out = self.output_activation(self.out(h1).squeeze())

        return out, [h1]


class MLP231(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super(MLP231, self).__init__()
        self.name = "MLP231"
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        self.h1 = nn.Linear(n_in, 2)
        self.h2 = nn.Linear(2, 3)
        self.out = nn.Linear(3, n_out)

    def forward(self, x):
        h1 = self.activation(self.h1(x))
        h2 = self.activation(self.h2(h1))
        out = self.output_activation(self.out(h2).squeeze())

        return out, [h1, h2]


class MLP321(nn.Module):
    def __init__(self, n_in=2, n_out=1):
        super(MLP321, self).__init__()
        self.name = "MLP321"
        self.activation = nn.ReLU()
        self.output_activation = nn.Sigmoid()

        self.h1 = nn.Linear(n_in, 3)
        self.h2 = nn.Linear(3, 2)
        self.out = nn.Linear(2, n_out)

    def forward(self, x):
        h1 = self.activation(self.h1(x))
        h2 = self.activation(self.h2(h1))
        out = self.output_activation(self.out(h2).squeeze())

        return out, [h1, h2]


# def train(model, EPOCHS = 10000):
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters())
#     losses = []
#     layer1 = []
#     layer2 = []
#     output = []

#     # Training loop
#     for epoch in range(EPOCHS):
#         model.train()
#         optimizer.zero_grad()

#         out, l2, l1 = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()

#         optimizer.zero_grad()

#         # Append relevant data
#         losses.append(to_np(loss))
#         layer1.append(to_np(l1))
#         layer2.append(to_np(l2))
#         output.append(to_np(out))
#         if epoch % 100 == 0:
#             print(f'Epoch {epoch}, Loss: {loss.item()}')
#     torch.save(model.state_dict(), 'model.pth')
#     np.save('losses.npy', losses)
#     np.save('input.npy', input)
#     np.save('target.npy', target)
#     np.save('layer1.npy', layer1)
#     np.save('layer2.npy', layer2)
#     np.save('output.npy', output)

# def load_model():
#     model = MLP()
#     try:
#         model.load_state_dict(torch.load('model.pth'))
#     except FileNotFoundError:
#         print("Model file not found. Training model.")
#         train(model)
#         model.load_state_dict(torch.load('model.pth'))
#     return model


if __name__ == "__main__":
    mod = MLP231()
    print(mod)
