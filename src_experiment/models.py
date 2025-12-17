import torch
from .utils import NeuralNet


# Small model

small_uniform_model = NeuralNet(input_size=2, hidden_sizes=[3,3,3], num_classes=1)

medium_uniform_model = NeuralNet(input_size=2, hidden_sizes=[5,5,5], num_classes=1)

increasing_model = NeuralNet(input_size=2, hidden_sizes=[3,5,7], num_classes=1)

decreasing_model = NeuralNet(input_size=2, hidden_sizes=[7,5,3], num_classes=1)

moons_models = {
    "small_uniform": small_uniform_model,
    "medium_uniform": medium_uniform_model,
    "increasing": increasing_model,
    "decreasing": decreasing_model,
    }


small_model_params = {
    "input_size": 2,
    "hidden_sizes": [3,3,3],
    "num_classes": 1
}

decreasing_model_params = {
    "input_size": 2,
    "hidden_sizes": [5,4,3],
    "num_classes": 1
}

def get_model(type="small", seed=None):
    match type:
        case "small":
            return NeuralNet(**small_model_params, seed=seed)
        case "decreasing":
            return NeuralNet(**decreasing_model_params, seed=seed)
        case _:
            raise ValueError