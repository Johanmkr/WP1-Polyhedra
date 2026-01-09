import matplotlib.pyplot as plt

def plot_training(res):
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


def plot_both_KL_IS(estimates, lw=2, super_title=""):
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
    MI = estimates["MI_KL"]
    IS = estimates["MI_IS"]
    
    # layers = df.columns[1:-1]  # skip first (epoch) and last

    fig, (ax0, ax1) = plt.subplots(nrows=2, ncols=1, figsize=(10, 7), sharex=True)

    for Q, ax, labels in zip([MI, IS], [ax0, ax1], ["KL", "IS"]):
        for layer in Q.columns[1:-1]:
            ax.plot(Q["epoch"], Q[layer], lw=lw, label=layer)
    
    ax0.grid(True)
    ax1.grid(True)
    ax1.set_xlabel("Epoch")
    ax0.set_ylabel("Kullback-Leibler MI")
    ax1.set_ylabel("Itakura-Saito MI")
    fig.suptitle(super_title)
    plt.legend()
    plt.tight_layout()
    plt.show()