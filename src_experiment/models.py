import torch
from .utils import NeuralNet


# Small model

small_uniform_model = NeuralNet(input_size=2, hidden_sizes=[3,3,3], num_classes=1)

small_uniform_model_long = NeuralNet(input_size=2, hidden_sizes=[3,3,3,3,3], num_classes=1)

medium_uniform_model = NeuralNet(input_size=2, hidden_sizes=[5,5,5], num_classes=1)

large_uniform_model = NeuralNet(input_size=2, hidden_sizes=[7,7,7], num_classes=1)

increasing_model = NeuralNet(input_size=2, hidden_sizes=[3,5,7], num_classes=1)

decreasing_model = NeuralNet(input_size=2, hidden_sizes=[7,5,3], num_classes=1)

moons_models = {
    "small_uniform": small_uniform_model,
    # "small_uniform_long": small_uniform_model_long,
    "medium_uniform": medium_uniform_model,
    "large_uniform": large_uniform_model,
    "increasing": increasing_model,
    "decreasing": decreasing_model,
    }