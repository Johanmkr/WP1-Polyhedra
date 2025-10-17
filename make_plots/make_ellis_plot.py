# import numpy as np
# import matplotlib.pyplot as plt
# import pathlib as pl
# import pickle
# import os, sys
# import pandas as pd
# # Get the parent directory
# parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# # Add parent directory to sys.path
# sys.path.append(parent_dir)
# import src.treenode as tn
# ground_path = pl.Path(parent_dir) / "state_dicts"


# params = {'legend.fontsize': 'x-large',
#           'figure.figsize': (12, 10),
#          'axes.labelsize': 'xx-large',
#          'axes.titlesize':'x-large',
#          'xtick.labelsize':'xx-large',
#          'ytick.labelsize':'xx-large'}
# plt.rcParams.update(params)

# def plot_ellis_data():
    
    
#     # Load the information theory data (pickle data)
#     with open(ground_path/"mm_sd"/"mi_information.pkl", "rb") as fp:
#         clean_data_lwd = pickle.load(fp)
#     with open(ground_path/"mm_ssd"/"mi_information.pkl", "rb") as fp:
#         noisy_data_lwd = pickle.load(fp)
    
#     clean_training = pd.read_pickle(ground_path/"mm_sd"/"training_data.pkl")
#     noisy_training = pd.read_pickle(ground_path/"mm_ssd"/"training_data.pkl")

#     # Load training data
    
#     fig, axes = plt.subplots(2, 2, sharex='col', sharey='row')
    
#     clean_axes = axes[:,0]
#     noisy_axes = axes[:,1]
    
    
#     # Plot MI
#     markers = ["o", "v", "*", ">"]
#     for ax, lwd in zip(axes[1], [clean_data_lwd, noisy_data_lwd]):
#         for i, (layer, data) in enumerate(lwd["layer_wise_data"].items()):
#             if layer in [0, "0"] or layer in [5, "5"]:
#                 continue
#             ax.plot(lwd["epochs"][::4], data[::4], ls="-", marker=markers[i-1], lw=1.5, markersize=10, label=f"Layer: {layer}")
            
#     # Plot training data
    
#     for ax1, train_data in zip(axes[0], [clean_training, noisy_training]):
#         train_loss = train_data["train_loss"]
#         train_accuracy = train_data["train_accuracy"]
#         test_loss = train_data["test_loss"]
#         test_accuracy = train_data["test_accuracy"]
        
#         ax1.plot(train_loss, label='Train Loss', ls='--', c='blue')
#         ax1.plot(test_loss, label='Test Loss', ls='-', c='blue')
#         # ax1.set_xlabel('Epochs')
#         # ax1.set_title(f'Loss and Accuracy: {exp_results["exp_name"][0]}')

#         ax2 = ax1.twinx()
#         ax2.plot(train_accuracy, label='Train Accuracy', ls='--', c='red')
#         ax2.plot(test_accuracy, label='Test Accuracy', ls='-', c='red')
#         counter = 0
#         if counter == 0:
#             ax2.set_yticks([])
#             counter += 1
#         else:
#             ax2.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9])
            
#     # ax.set_title(titles[exp_name])
#     # ax.set_xlabel("Epoch")
#     # ax.set_ylabel(r"$MI(L,Y)$")
#     plt.legend()
#     plt.tight_layout()
#     # plt.savefig(pl.Path(parent_dir)/"figures"/f"MI_{layer_wise_dict['exp_name']}.pdf")
#     plt.show()
    
# if __name__ == "__main__":
#     plot_ellis_data()


import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import pickle
import os, sys
import pandas as pd

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import src.treenode as tn

ground_path = pl.Path(parent_dir) / "state_dicts"

params = {
    'legend.fontsize': 'x-large',
    'figure.figsize': (12, 10),
    'axes.labelsize': 'xx-large',
    'axes.titlesize': 'xx-large',
    'xtick.labelsize': 'xx-large',
    'ytick.labelsize': 'xx-large'
}
plt.rcParams.update(params)


def plot_ellis_data():
    # --- Load data ---
    with open(ground_path / "mm_sd" / "mi_information.pkl", "rb") as fp:
        clean_data_lwd = pickle.load(fp)
    with open(ground_path / "mm_ssd" / "mi_information.pkl", "rb") as fp:
        noisy_data_lwd = pickle.load(fp)

    clean_training = pd.read_pickle(ground_path / "mm_sd" / "training_data.pkl")
    noisy_training = pd.read_pickle(ground_path / "mm_ssd" / "training_data.pkl")

    # --- Create figure ---
    fig, axes = plt.subplots(
        2, 2,
        sharex='col', sharey='row',
        figsize=(18, 7),
        gridspec_kw={'wspace': 0.0, 'hspace': 0.0}
    )

    # clean_axes = axes[:, 0]
    # noisy_axes = axes[:, 1]

    # --- Top row: Training data with shared right y-axis ---
    for ax1, train_data in zip(axes[0], [clean_training, noisy_training]):
        ax2 = ax1.twinx()

        train_loss = train_data["train_loss"]
        train_accuracy = train_data["train_accuracy"]
        test_loss = train_data["test_loss"]
        test_accuracy = train_data["test_accuracy"]

        ax1.plot(train_loss, label='Train Loss', ls='--', c='blue')
        ax1.plot(test_loss, label='Test Loss', ls='-', c='blue')
        ax2.plot(train_accuracy, label='Train Accuracy', ls='--', c='red')
        ax2.plot(test_accuracy, label='Test Accuracy', ls='-', c='red')
        ax1.grid(True)

        # Hide the *entire* twin y-axis on the left subplot
        if ax1 is axes[0, 0]:
            print("Axis")
            # Disable all ticks and labels on the right
            ax2.tick_params(axis='y', which='both',
                            right=False, labelright=False)

            # Hide the right spine completely
            ax2.spines['right'].set_visible(False)

            # Tell constrained_layout / tight_layout to ignore this axis
            ax2.set_frame_on(False)
            for spine in ax2.spines.values():
                spine.set_visible(False)

            # Extra insurance: remove tick locators and formatters
            ax2.yaxis.set_major_locator(plt.NullLocator())
            ax2.yaxis.set_minor_locator(plt.NullLocator())
            ax2.yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax1.legend(loc="center right", frameon=False)
            ax2.legend(loc="lower right", frameon=False)
            ax2.set_ylabel('Accuracy')


    # Make right y-axis of top row shared
    # (link top-right twin y-axis to the top-left’s twin y-axis)
    # axes[0, 0].right_ax = axes[0, 0].twinx()
    # axes[0, 1].right_ax = axes[0, 0].right_ax

    # --- Bottom row: Mutual Information ---
    markers = ["o", "v", "*", ">"]
    for ax, lwd in zip(axes[1], [clean_data_lwd, noisy_data_lwd]):
        for i, (layer, data) in enumerate(lwd["layer_wise_data"].items()):
            if layer in [0, "0"] or layer in [5, "5"]:
                continue
            ax.plot(
                lwd["epochs"][::4],
                data[::4],
                ls="-",
                marker=markers[i - 1],
                lw=1.5,
                markersize=10,
                label=r"$L={layer}$".format(layer=layer)
            )
        ax.grid(True)

    # --- Titles & Labels ---
    axes[0, 0].set_title("Clean Data")
    axes[0, 1].set_title("Noisy Data")

    axes[0, 0].set_ylabel("Loss")
    axes[1, 0].set_ylabel(r"$MI(L,Y)$")
    axes[1, 0].set_xlabel("Epochs")
    axes[1, 1].set_xlabel("Epochs")

    # --- Legends per row ---
    # Top row: combine both loss & accuracy handles
    # handles_top, labels_top = [], []
    # for ax in [axes[0, 1], axes[0, 1].twinx()]:
    #     h, l = ax.get_legend_handles_labels()
    #     handles_top += h
    #     labels_top += l
    # axes[0, 0].legend(handles_top, labels_top, loc="upper left", frameon=False)

    # Bottom row: single legend for MI
    handles_bottom, labels_bottom = axes[1, 1].get_legend_handles_labels()
    axes[1, 1].legend(handles_bottom, labels_bottom, loc="upper left", frameon=False)

    # --- Layout tweaks ---
    plt.tight_layout()
    plt.savefig(pl.Path(parent_dir)/"figures"/f"loss_acc_and_MI.pdf")
    plt.show()


if __name__ == "__main__":
    plot_ellis_data()
