import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np
import torch.nn as nn
import torch


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNet, self).__init__()
        self.hidden_sizes = hidden_sizes

        for i in range(len(hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            if i == 0:
                setattr(self, layer_name, nn.Linear(input_size, hidden_sizes[i]))
                setattr(self, relu_name, nn.ReLU())
            else:
                setattr(
                    self, layer_name, nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
                )
                setattr(self, relu_name, nn.ReLU())

        output_layer_name = f"l{len(hidden_sizes) + 1}"

        setattr(self, output_layer_name, nn.Linear(hidden_sizes[-1], num_classes))

    def forward(self, x):
        out = x
        for i in range(len(self.hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            out = getattr(self, layer_name)(out)
            out = getattr(self, relu_name)(out)

        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        out = getattr(self, output_layer_name)(out)
        return out

class DataGenerator:
    def __init__(self, n=100):
        self.n = n
    def convex(n=100):
        x_1 = np.linspace(-1, 1, n, dtype="float32")
        x_2 = np.linspace(-1, 1, n, dtype="float32")
        x_1, x_2 = np.meshgrid(x_1, x_2)
        x_1 = np.reshape(x_1, (n**2,1))
        x_2 = np.reshape(x_2, (n**2,1))
        
        y = np.asarray([i**2+j**2-2/3 for i, j in zip(x_1, x_2)])
        x = np.concatenate((x_1,x_2), axis = 1)
        return x, y

    def franke(n=100):
        x = np.linspace(0, 1, n)
        y = np.linspace(0, 1, n)
        x_1, x_2 = np.meshgrid(x, y)
        x_1 = np.reshape(x_1, (n**2, 1))
        x_2 = np.reshape(x_2, (n**2, 1))
        y = 0.75 * np.exp(-(9 * x_1 - 2) ** 2 / 4.0 - (9 * x_2 - 2) ** 2 / 4.0) + \
            0.75 * np.exp(-(9 * x_1 + 1) ** 2 / 49.0 - (9 * x_2 + 1) ** 2 / 10.0) + \
            0.5 * np.exp(-(9 * x_1 - 7) ** 2 / 4.0 - (9 * x_2 - 3) ** 2 / 49.0) - \
            0.2 * np.exp(-(9 * x_1 - 4) ** 2 - (9 * x_2 - 7) ** 2)
        x = np.concatenate((x_1, x_2), axis=1)
        return x, y



if __name__ == "__main__":
    pass
