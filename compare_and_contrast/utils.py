import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

torch.manual_seed(42)

def generate_nn_csv(layer_sizes, filename="dummy_weights.csv"):
    """
    layer_sizes: List of ints, e.g., [10, 20, 20, 5] 
    (10 input, two hidden layers of 20, 5 output)
    """
    data = []

    for i in range(len(layer_sizes) - 1):
        in_dim = layer_sizes[i]
        out_dim = layer_sizes[i+1]
        
        # 1. Generate Random Weights (Matrix: out_dim x in_dim)
        # Using Kaiming Uniform initialization logic for "realism"
        weights = torch.randn(out_dim, in_dim) * (2/in_dim)**0.5
        
        for r in range(out_dim):
            for c in range(in_dim):
                data.append({
                    "layer": f"layer_{i+1}",
                    "type": "weight",
                    "coord": f"({r},{c})",
                    "value": weights[r, c].item()
                })
        
        # 2. Generate Random Biases (Vector: out_dim)
        biases = torch.randn(out_dim) * 0.01
        
        for r in range(out_dim):
            data.append({
                "layer": f"layer_{i+1}",
                "type": "bias",
                "coord": f"({r})",
                "value": biases[r].item()
            })

    # Convert to DataFrame and save
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    print(f"Successfully saved dummy weights to {filename}")







# --- 1. Data and State Setup ---



def to_tuple(array):
    """Converts a 1D numpy array to a hashable tuple."""
    return tuple(array.ravel())

# --- 2. Helper Functions ---

def clip_to_bounds(x, y, x_min, x_max, y_min, y_max):
    return (x_min <= x <= x_max) and (y_min <= y <= y_max)

def plot_hyperplane(ax, w, b, color, label, offset=0.15, num_marks=6, fac=5):
    """Plot a hyperplane with +/- markers showing the activation direction."""
    # Check if the line is not perfectly vertical
    x_min, x_max = -fac, fac
    y_min, y_max = -fac, fac

    if abs(w[1]) > 1e-6:
        x_vals = np.linspace(x_min, x_max, 1000)
        y_vals = -(w[0]*x_vals + b)/w[1]
        mask = (y_vals >= y_min) & (y_vals <= y_max)
        
        if np.any(mask):
            x_line, y_line = x_vals[mask], y_vals[mask]
            ax.plot(x_line, y_line, color=color, label=label, linewidth=1.5)

            # Normal vector for marker direction
            n = w / np.linalg.norm(w)
            
            # Interpolate points along the visible segment for markers
            ds = np.sqrt(np.diff(x_line)**2 + np.diff(y_line)**2)
            s = np.concatenate(([0], np.cumsum(ds)))
            s_targets = np.linspace(s[0] + 5, s[-1] - 5, num_marks)
            x_marks = np.interp(s_targets, s, x_line)
            y_marks = np.interp(s_targets, s, y_line)

            for xm, ym in zip(x_marks, y_marks):
                # Calculate positive and negative side positions
                for sign, direction in [('+', 1), ('-', -1)]:
                    xt, yt = xm + direction * offset * n[0], ym + direction * offset * n[1]
                    if clip_to_bounds(xt, yt, x_min, x_max, y_min, y_max):
                        ax.text(xt, yt, sign, color=color, ha='center', va='center', 
                                fontsize=14, weight='bold',
                                path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    else:
        # Vertical line case (w[1] is 0)
        x_line = -b/w[0]
        if x_min <= x_line <= x_max:
            ax.axvline(x_line, color=color, label=label, linewidth=1.5)
            n = w / np.linalg.norm(w)
            y_marks = np.linspace(y_min + 10, y_max - 10, num_marks)
            for ym in y_marks:
                for sign, direction in [('+', 1), ('-', -1)]:
                    xt = x_line + direction * offset * n[0]
                    if clip_to_bounds(xt, ym, x_min, x_max, y_min, y_max):
                        ax.text(xt, ym, sign, color=color, ha='center', va='center', 
                                fontsize=14, weight='bold',
                                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

# --- 3. Plotting Execution ---







if __name__=="__main__":
    generate_nn_csv([2, 5,5,5, 5])
    
    