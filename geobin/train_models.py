import numpy as np
import torch
import torch.nn as nn
from tqdm import trange
import pathlib as pl
import matplotlib.pyplot as plt
import pandas as pd

import os, sys
# Get the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Add parent directory to sys.path
sys.path.append(parent_dir)
from src import treenode as tn
from src import utils

from sklearn.datasets import make_moons



params = {'legend.fontsize': 'x-large',
          'figure.figsize': (8, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(params)

# Make model

small_model = utils.NeuralNet(input_size=2, hidden_sizes=[3,3], num_classes=1)
large_model = utils.NeuralNet(input_size=2, hidden_sizes=[3,3,3,3,3], num_classes=1)

X_train, y_train = make_moons(n_samples=10000, noise=0.05, random_state=1)
X_test, y_test = make_moons(n_samples=2500, noise=0.05, random_state=11)



X_train_noise, y_train_noise = make_moons(n_samples=10000, noise=2, random_state=1)
X_test_noise, y_test_noise = make_moons(n_samples=2500, noise=2, random_state=11)



EPOCHS = 10000
criterion = nn.BCEWithLogitsLoss()

ground_path = pl.Path(parent_dir) / "state_dicts"

def train_model(experiment, SAVE_STATES = False, save_everyth_epoch = 100):
    exp_name = experiment["exp_name"]
    model = experiment["model"]
    data = experiment["data"]
    train_data = torch.tensor(data[0][0], dtype=torch.float32)
    train_target = torch.tensor(data[0][1], dtype=torch.float32)
    test_data = torch.tensor(data[1][0], dtype=torch.float32)
    test_target = torch.tensor(data[1][1], dtype=torch.float32)
    
    # Check savepath
    savedir = ground_path / exp_name
    if not savedir.is_dir():
        os.makedirs(savedir)
    
    train_loss, train_accuracy = [], []
    test_loss, test_accuracy = [], []
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,momentum=0.1)
    for epoch in trange(EPOCHS):
        model.train()
        optimizer.zero_grad()
        
        out = model(train_data)
        loss = criterion(out, train_target.unsqueeze(1))
        loss.backward()
        train_loss.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
        preds = torch.sigmoid(out) > 0.5
        accuracy = (preds.squeeze() == train_target).float().mean().item()
        train_accuracy.append(accuracy)
        
        model.eval()
        with torch.no_grad():
            out = model(test_data)
            loss = criterion(out, test_target.unsqueeze(1))
            preds = torch.sigmoid(out) > 0.5
            accuracy = (preds.squeeze() == test_target).float().mean().item()
            test_loss.append(loss.item())
            test_accuracy.append(accuracy)
        if SAVE_STATES and (epoch % save_everyth_epoch == 0 or epoch == EPOCHS-1):
            torch.save(model.state_dict(), savedir / f"{exp_name}_epoch{epoch}.pth")
        
    experiment_results = {
        "exp_name": exp_name,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy
    }
    if SAVE_STATES:
        frame = pd.DataFrame.from_dict(experiment_results)
        frame.to_pickle(ground_path/exp_name/"training_data.pkl")
    return experiment_results

def plot_data():
    fig, ax = plt.subplots(figsize=(7,7))
    
    ax.scatter(X_train_noise[:,0], X_train_noise[:,1], c=y_train_noise, s=15)
    ax.scatter(X_train[:,0], X_train[:,1], c=y_train*-2, s=15)
    plt.show()


def plot_results(exp_results):
    fig, ax1 = plt.subplots(figsize=(7, 7))
    train_loss = exp_results["train_loss"]
    train_accuracy = exp_results["train_accuracy"]
    test_loss = exp_results["test_loss"]
    test_accuracy = exp_results["test_accuracy"]
    
    ax1.plot(train_loss, label='Train Loss', ls='--', c='blue')
    ax1.plot(test_loss, label='Test Loss', ls='-', c='blue')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Loss and Accuracy: {exp_results["exp_name"][0]}')
    ax1.legend(loc="lower center")

    ax2 = ax1.twinx()
    ax2.plot(train_accuracy, label='Train Accuracy', ls='--', c='red')
    ax2.plot(test_accuracy, label='Test Accuracy', ls='-', c='red')
    ax2.set_ylabel('Accuracy')
    ax2.legend(loc="upper center")
    plt.savefig(pl.Path(parent_dir)/"figures"/f"training_data_{exp_results['exp_name'][0]}.png")
    plt.show()
    
def load_and_plot_training_data(exp_name):
    loadpath = ground_path/exp_name/"training_data.pkl"
    frame = pd.read_pickle(loadpath)
    plot_results(frame)
    

            
# Define experiments as dicts:
# sm = small model
# lm = large model
# sd = smooth data
# nd = noisy data
smooth_data = [[X_train, y_train], [X_test, y_test]]
noisy_data = [[X_train_noise, y_train_noise], [X_test_noise, y_test_noise]]

# Experiment 1 - Small model, smooth data
exp1 = {
    "exp_name": "sm_sd",
    "model": small_model,
    "data": smooth_data,
}

# Experiment 2 - Small model, noisy data
exp2 = {
    "exp_name": "sm_nd",
    "model": small_model,
    "data": noisy_data,
}

# Experiment 3 - Large model, smooth data
exp3 = {
    "exp_name": "lm_sd",
    "model": large_model,
    "data": smooth_data,
}

# Experiment 4 - Large model, noisy data
exp4 = {
    "exp_name": "lm_nd",
    "model": large_model,
    "data": noisy_data,
}

# Experiment 5 - Medium model, smooth data
exp5 = {
    "exp_name": "mm_sd",
    "model": utils.NeuralNet(input_size=2, hidden_sizes=[3,3,3,3], num_classes=1),
    "data": [make_moons(n_samples=10000, noise=0.05, random_state=1), make_moons(n_samples=2500, noise=0.05, random_state=11)]
}

# Experiment 6 - Medium model, semi-smooth-data
exp6 = {
    "exp_name": "mm_ssd",
    "model": utils.NeuralNet(input_size=2, hidden_sizes=[3,3,3,3], num_classes=1),
    "data": [make_moons(n_samples=10000, noise=0.25, random_state=1), make_moons(n_samples=2500, noise=0.25, random_state=11)]
}

# Experiment 7 - Medium model, noisy data 
exp7 = {
    "exp_name": "mm_nd",
    "model": utils.NeuralNet(input_size=2, hidden_sizes=[3,3,3,3], num_classes=1),
    "data": [make_moons(n_samples=10000, noise=0.5, random_state=1), make_moons(n_samples=2500, noise=0.5, random_state=11)]
}

def plot_training_moon_data(list_of_exp_dicts):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots()
    # colors=["r", "g", "b"]
    for dictionary in list_of_exp_dicts:
        data = dictionary["data"][0] # Plot training data only
        class0data = data[0][data[1]==0]
        class1data = data[0][data[1]==1]
        ax.scatter(data[0][::25,0], data[0][::25,1],c=data[1][::25], marker="o")
        # ax.scatter(class1data[::25,0], class1data[::25,1],c=color, marker="x")
        
    # Plot range
    x_min, x_max = -1.5, 2.5
    y_min, y_max = -1.5, 1.5
    
    np.random.seed(3)
    W1 = np.random.randn(3,2)
    b1 = np.random.randn(3,1)

    colors = plt.cm.tab10.colors  # distinct colors for each hyperplane

    def clip_to_bounds(x, y, x_min, x_max, y_min, y_max):
        """Check if point lies inside plot boundaries."""
        return (x_min <= x <= x_max) and (y_min <= y <= y_max)

    for i in range(W1.shape[0]):
        w = W1[i]
        b = b1[i].item()
        color = "k"
        offset = 0.15  # closer to the line than before

        if abs(w[1]) > 1e-6:
            # Line within bounds
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = -(w[0]*x_vals + b)/w[1]

            mask = (y_vals >= y_min) & (y_vals <= y_max)
            x_line = x_vals[mask]
            y_line = y_vals[mask]

            hp_line = ax.plot(x_line, y_line, color=color)

            # Normal vector
            n = w / np.linalg.norm(w)

            # Cumulative arc length
            ds = np.sqrt(np.diff(x_line)**2 + np.diff(y_line)**2)
            s = np.concatenate(([0], np.cumsum(ds)))

            # Evenly spaced markers
            num_marks = 6
            s_targets = np.linspace(0, s[-1], num_marks)
            x_marks = np.interp(s_targets, s, x_line)
            y_marks = np.interp(s_targets, s, y_line)

            # Place + and - signs with black outline
            for xm, ym in zip(x_marks, y_marks):
                xp, yp = xm + offset*n[0], ym + offset*n[1]
                xm_, ym_ = xm - offset*n[0], ym - offset*n[1]
                if clip_to_bounds(xp, yp, x_min, x_max, y_min, y_max):
                    plt.text(xp, yp, '+', color=color, ha='center', va='center', fontsize=12,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
                if clip_to_bounds(xm_, ym_, x_min, x_max, y_min, y_max):
                    plt.text(xm_, ym_, '-', color=color, ha='center', va='center', fontsize=12,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])

        else:
            # Vertical line: x = -b/w[0]
            x_line = -b/w[0]
            if x_min <= x_line <= x_max:
                hp_line = ax.axvline(x_line, color=color)

            n = w / np.linalg.norm(w)
            y_marks = np.linspace(y_min, y_max, 6)

            for ym in y_marks:
                xp, yp = x_line + offset*n[0], ym
                xm_, ym_ = x_line - offset*n[0], ym
                if clip_to_bounds(xp, yp, x_min, x_max, y_min, y_max):
                    plt.text(xp, yp, '+', color=color, ha='center', va='center', fontsize=12,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])
                if clip_to_bounds(xm_, ym_, x_min, x_max, y_min, y_max):
                    plt.text(xm_, ym_, '-', color=color, ha='center', va='center', fontsize=12,
                            path_effects=[pe.withStroke(linewidth=1.2, foreground="black")])

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"$x_2$")
    # plt.title("Example of hyperplanes and data points")
    # plt.legend()
    plt.grid(False)
    plt.tight_layout()
    plt.savefig(pl.Path(parent_dir)/"figures"/f"data_and_hyperplanes.pdf")
    plt.show()
        




if __name__=="__main__":
    # res1 = train_model(exp1, SAVE_STATES=True)
    # res2 = train_model(exp2, SAVE_STATES=True)
    # res3 = train_model(exp3, SAVE_STATES=True)
    # res4 = train_model(exp4, SAVE_STATES=True)
    
    
    # plot_results(res1)
    # plot_results(res2)
    # plot_results(res3)
    # plot_results(res4)
    
    # for exp_name in ["sm_sd", "sm_nd", "lm_sd", "lm_nd"]:
    #     load_and_plot_training_data(exp_name)
    
    # res5 = train_model(exp5, SAVE_STATES=True)
    # res6 = train_model(exp6, SAVE_STATES=True)
    # res7 = train_model(exp7, SAVE_STATES=True)
    # plot_results(res5)
    # plot_results(res6)
    # plot_results(res7)
    
    # for exp_name in ["mm_sd","mm_ssd"]:
    #     load_and_plot_training_data(exp_name)
    
    
    
    # plot_training_moon_data([exp5])
    pass