from src import treenode as tn
import numpy as np
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, classification=False):
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
        # setattr(self, "output_activation", nn.Softmax(dim=num_classes - 1) if classification else nn.Identity())
        
    def forward(self, x):
        out = x
        for i in range(len(self.hidden_sizes)):
            layer_name = f"l{i + 1}"
            relu_name = f"relu{i + 1}"
            out = getattr(self, layer_name)(out)
            out = getattr(self, relu_name)(out)

        output_layer_name = f"l{len(self.hidden_sizes) + 1}"
        out = getattr(self, output_layer_name)(out)
        # out = getattr(self, "output_activation")(out)
        # if len(out.shape) == 1:
        #     out = out.unsqueeze(0)
        # elif len(out.shape) == 2 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        # elif len(out.shape) == 3 and out.shape[0] == 1:
        #     out = out.squeeze(0)
        return out
if __name__ == "__main__":
    # Make model
    model = NeuralNet(input_size=2, num_classes=1, hidden_sizes=[4,6,5,3])
    tree = tn.RegionTree(model.state_dict())
    tree.build_tree()
    root = tree.get_root()
    print("Tree built successfully")
