from .utils import NeuralNet

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

small_model_params_wbc = {
    "input_size": 30,
    "hidden_sizes": [3,3,3],
    "num_classes": 1
}

decreasing_model_params_wbc = {
    "input_size": 30,
    "hidden_sizes": [5,4,3],
    "num_classes": 1
}


def get_model(type="small_moon", dropout=0.0, seed=None):
    match type:
        case "small_moon":
            return NeuralNet(**small_model_params, dropout=dropout, seed=seed)
        case "decreasing_moon":
            return NeuralNet(**decreasing_model_params, dropout=dropout, seed=seed)
        case "small_wbc":
            return NeuralNet(**small_model_params_wbc, dropout=dropout, seed=seed)
        case "decreasing_wbc":
            return NeuralNet(**decreasing_model_params_wbc, dropout=dropout, seed=seed)
        case _:
            raise ValueError