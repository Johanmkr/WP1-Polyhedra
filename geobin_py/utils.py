import numpy as np

def find_hyperplanes(state_dict):
    weights = []
    biases = []
    hyperplanes = []
    for key, val in state_dict.items():
        if "weight" in key:
            weights.append(val)
        elif "bias" in key:
            biases.append(val)
    for W, b in zip(weights, biases):
        hyperplanes.append(np.hstack((W, b.reshape(-1,1))))
    return hyperplanes

def to_tuple(array):
    return tuple(array.ravel())
