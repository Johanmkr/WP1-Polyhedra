import numpy as np
import re

def find_hyperplanes(state_dict):
    # 1. Identify keys that contain "weight"
    keys_weights = [k for k in state_dict.keys() if "weight" in k]

    # 2. Sort keys numerically so layer_10 comes after layer_2
    def extract_layer_idx(k):
        m = re.search(r"(\d+)", k)
        return int(m.group(1)) if m else -1

    sorted_keys = sorted(keys_weights, key=extract_layer_idx)

    weights = []
    biases = []

    for k_w in sorted_keys:
        # 3. Create the corresponding bias key name
        k_b = k_w.replace("weight", "bias")
        
        # 4. Extract and convert to Float64 numpy arrays
        # We use .astype(float) to match Julia's Float64
        W = np.array(state_dict[k_w]).astype(float)
        b = np.array(state_dict[k_b]).astype(float)
        
        weights.append(W)
        biases.append(b)

    # 5. Return as a tuple of lists, exactly like the Julia return weights, biases
    return weights, biases

def to_tuple(array):
    return tuple(array.ravel())
