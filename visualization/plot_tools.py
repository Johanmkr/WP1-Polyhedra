import os
import math
import pandas as pd
import matplotlib.pyplot as plt

def plot_training(res, title=None):
    ax = res[["train_loss", "test_loss"]].plot(
        figsize=(10, 10),
        color=["blue", "blue"],
        style=["-", "--"],
        ylabel="Loss",
    )

    ax2 = res[["train_accuracy", "test_accuracy"]].plot(
        ax=ax,
        secondary_y=True,
        color=["red", "red"],
        style=["-", "--"],
    )

    ax.set_xlabel("Epochs")
    ax2.set_ylabel("Accuracy")

    # Fix legends (pandas splits them)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax.set_title(title if title is not None else "Training results")
    plt.show()
    
def plot_results_on_ax(res, ax, title=None):
    res[["train_loss", "test_loss"]].plot(
        ax=ax, style=["-", "--"], legend=False
    )

    ax_acc = ax.twinx()
    res[["train_accuracy", "test_accuracy"]].plot(
        ax=ax_acc, style=["-", "--"], legend=False
    )

    ax.set(xlabel="Epochs", ylabel="Loss", title=title)
    ax_acc.set_ylabel("Accuracy")

    return ax, ax_acc


def plot_all_results(results, titles=None, run_number=0):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.ravel()

    acc_axes = []

    for i, res in enumerate(results):
        _, ax_acc = plot_results_on_ax(
            res,
            axes[i],
            None if titles is None else titles[i],
        )
        acc_axes.append(ax_acc)

    # Legend from first subplot (both y-axes)
    lines = axes[0].lines + acc_axes[0].lines
    labels = ["Train loss", "Test loss", "Train acc", "Test acc"]

    fig.legend(lines, labels, loc="upper left", ncol=4)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.suptitle(f"Run number: {run_number}")
    plt.show()


def plot_all_quantities(estimates, lw=2, super_title=""):
    """
    Plot all layers except the last one on the same figure as functions of epoch.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'epoch' and layers like 'l1', 'l2', ...
    lw : float
        Line width for the curves.
    """
    # All columns except 'epoch' and the last layer
    MI = estimates["Kullback-Leibler"]
    IS = estimates["Itakura-Saito"]
    HW = estimates["H(W)"]
    HYW = estimates["H(Y|W)"]
    
    quantities = [MI, IS, HW, HYW]
    
    # layers = df.columns[1:-1]  # skip first (epoch) and last

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 7), sharex=True)

    for Q, ax, label in zip(quantities, axes.flat, ["KL", "IS", "H(W)", "H(Y|W)"]):
        for layer in Q.columns[1:-1]:
            ax.plot(Q["epoch"], Q[layer], ls="-", lw=lw, marker="o", label=layer)
            ax.set_title(label, pad=15)
            ax.grid(True)
            ax.legend(loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=8,
            frameon=False,)
            
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 0].set_xlabel("Epoch")
    # axes[0, 0].set_ylabel("Quantity value")
    # axes[1, 0].set_ylabel("Quantity value")
    
    fig.suptitle(super_title, fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.show()
    


def plot_training_sweep(
    vary,
    values,
    *,
    arch,
    dropout,
    noise,
    run_number,
    path_fn,
):
    """
    Create a subplot figure sweeping over one parameter.

    Parameters
    ----------
    vary : str
        One of {"arch", "dropout", "noise", "run_number"}
    values : list
        Values of the parameter to vary
    arch, dropout, noise, run_number :
        Fixed parameters (except the one being varied)
    moon_path_fn : callable
        Function moon_path(arch, dropout, noise, run_number)
    """
    assert vary in {"arch", "dropout", "noise", "run_number"}

    n = len(values)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5 * ncols, 4 * nrows),
        squeeze=False,
        sharex=True,
        # sharey=True,
    )

    for idx, val in enumerate(values):
        r, c = divmod(idx, ncols)
        ax_loss = axes[r][c]

        # Set parameters
        params = dict(
            arch=arch,
            dropout=dropout,
            noise=noise,
            run_number=run_number,
        )
        params[vary] = val

        path = path_fn(**params)
        df = pd.read_csv(os.path.join(path, "run_summary.csv"))

        epochs = df.index.to_numpy()

        # Loss
        ax_loss.plot(epochs, df["train_loss"], color="blue", linestyle="-", label="Train Loss")
        ax_loss.plot(epochs, df["test_loss"], color="blue", linestyle="--", label="Test Loss")
        ax_loss.set_xlabel("Epoch")
        ax_loss.set_ylabel("Loss")
        ax_loss.set_title(f"{vary} = {val}", pad=15)
        ax_loss.grid(True, alpha=0.3)

        # Accuracy
        ax_acc = ax_loss.twinx()
        ax_acc.plot(epochs, df["train_accuracy"], color="red", linestyle="-", label="Train Acc")
        ax_acc.plot(epochs, df["test_accuracy"], color="red", linestyle="--", label="Test Acc")
        ax_acc.set_ylabel("Accuracy")

        # Legend (only once per subplot)
        lines_1, labels_1 = ax_loss.get_legend_handles_labels()
        lines_2, labels_2 = ax_acc.get_legend_handles_labels()
        ax_loss.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=4,
        fontsize=8,
        frameon=False,
    )


    # Remove empty subplots if any
    for idx in range(n, nrows * ncols):
        r, c = divmod(idx, ncols)
        fig.delaxes(axes[r][c])

    def _format_fixed_params(vary, arch, dropout, noise, run_number):
        params = {
            "arch": arch,
            "dropout": dropout,
            "noise": noise,
            "run_number": run_number,
        }

        fixed = {k: v for k, v in params.items() if k != vary}
        return ", ".join(f"{k}={v}" for k, v in fixed.items())

    main_title = f"Training sweep over {vary}"
    subtitle = f"Fixed parameters: {_format_fixed_params(vary, arch, dropout, noise, run_number)}"
    fig.suptitle(main_title + "\n" + subtitle, fontsize=16, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    
    
def plot_average_KL(frame, noise=0.0, title="", **kwargs):
    
    fig, ax = plt.subplots(figsize=(5,4))
    for layer in frame.columns[1:-1]:
            ax.plot(frame["epoch"], frame[layer], ls="-", lw=2, marker="o", label=layer, **kwargs)
            ax.set_title(title, pad=15)
            ax.grid(True)
            ax.legend(loc="lower center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=4,
            fontsize=8,
            frameon=False,)
            
    plt.show()
    
