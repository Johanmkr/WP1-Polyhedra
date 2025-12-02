import copy
import pathlib as pl
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
import pandas as pd


# ================================================================
# Utility Functions
# ================================================================

def slope(values: np.ndarray, window: int = 5) -> Optional[float]:
    """
    Compute the slope between the mean of the most recent `window`
    values and the previous `window` values.

    Parameters
    ----------
    values : np.ndarray
        Array of recorded metric values.
    window : int, optional
        Size of the window over which to compute means.

    Returns
    -------
    float or None
        Returns the slope, or None if not enough elements exist.
    """
    if len(values) < 2 * window:
        return None

    recent = np.mean(values[-window:])
    previous = np.mean(values[-2 * window: -window])
    return (recent - previous) / window


def has_plateaued(values: np.ndarray, window: int = 5, tolerance: float = 0.01) -> bool:
    """
    Determine if a curve has plateaued using slope magnitude.

    Parameters
    ----------
    values : np.ndarray
        Sequence of metric values over training.
    window : int
        Window size used for computing slope.
    tolerance : float
        Upper bound on slope magnitude.

    Returns
    -------
    bool
        True if slope magnitude is below tolerance.
    """
    s = slope(values, window)
    return s is not None and abs(s) < tolerance


def is_stable(values: np.ndarray, tolerance: float = 0.03) -> bool:
    """
    Check whether recent values show small variance.

    Parameters
    ----------
    values : np.ndarray
        Array of values (e.g., accuracy).
    tolerance : float
        Threshold relative to the mean for stability.

    Returns
    -------
    bool
        True if the last 5 values have low variance.
    """
    if len(values) < 5:
        return False

    recent = values[-5:]
    mean = np.mean(recent)
    return np.std(recent) < tolerance * (mean + 1e-8)


# ================================================================
# Convergence & Local Minimum Detection
# ================================================================

def determine_convergence_and_minimum(
    train_loss: np.ndarray,
    val_loss: np.ndarray,
    train_acc: np.ndarray,
    val_acc: np.ndarray,
    min_acceptable_acc: float = 0.75
) -> Dict[str, Any]:
    """
    Analyze training curves to determine:

    1. Whether training converged.
    2. Whether the model is likely stuck in a poor local minimum.

    Parameters
    ----------
    train_loss : np.ndarray
        Training loss over epochs.
    val_loss : np.ndarray
        Validation loss over epochs.
    train_acc : np.ndarray
        Training accuracy over epochs.
    val_acc : np.ndarray
        Validation accuracy over epochs.
    min_acceptable_acc : float, optional
        Threshold below which model is considered poorly converged.

    Returns
    -------
    dict
        Contains:
        - converged
        - stability
        - loss_plateau
        - acc_plateau
        - overfitting
        - local_minimum
        - globalish_minimum
        - reasons (list)
    """

    results: Dict[str, Any] = {}

    # Plateaus
    loss_plateau = (
        has_plateaued(train_loss, 5, 0.005) and
        has_plateaued(val_loss, 5, 0.005)
    )
    acc_plateau = (
        has_plateaued(train_acc, 5, 0.002) and
        has_plateaued(val_acc, 5, 0.002)
    )

    stability = is_stable(train_acc) and is_stable(val_acc)

    # Overfitting detection: training improves while validation worsens
    overfitting = (
        train_loss[-1] < train_loss[-2] and
        val_loss[-1] > val_loss[-2]
    )

    converged = loss_plateau and acc_plateau and stability and not overfitting

    # --------- Local minimum detection ---------
    local_minimum = False
    reasons: List[str] = []

    # 1. Accuracy too low
    mean_final_acc = np.mean(val_acc[-5:])
    accurate = mean_final_acc >= min_acceptable_acc
    if not accurate:
        local_minimum = True
        reasons.append("Final validation accuracy below acceptable threshold.")

    # 2. Early stagnation
    if len(train_loss) > 15 and has_plateaued(train_loss, 10, 0.01):
        if np.argmin(train_loss) < len(train_loss) * 0.3:
            local_minimum = True
            reasons.append("Loss plateaued too early (early stagnation).")

    # 3. Insufficient improvement
    improvement = train_loss[0] - train_loss[-1]
    if improvement < 0.1 * train_loss[0]:
        local_minimum = True
        reasons.append("Insufficient improvement across training.")

    # 4. Validation loss relatively high
    if val_loss[-1] > np.median(val_loss) * 0.95:
        local_minimum = True
        reasons.append("Validation loss remains high.")

    results.update({
        "converged": converged,
        "stability": stability,
        "loss_plateau": loss_plateau,
        "acc_plateau": acc_plateau,
        "overfitting": overfitting,
        "local_minimum": local_minimum,
        "accurate": accurate,
        "globalish_minimum": converged and not local_minimum,
        "reasons": reasons
    })

    return results


# ================================================================
# Training Loop
# ================================================================

def train_model(
    model: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    savepath: Optional[pl.Path] = None,
    experiment_name: Optional[str] = None,
    SAVE_STATES: bool = False,
    save_everyth_epoch: int = 50,
    RETURN_STATES: bool = False
) -> Any:
    """
    Train a PyTorch model, record training curves, optionally save
    model snapshots, and perform convergence analysis.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    train_data : DataLoader
        Training dataset.
    test_data : DataLoader
        Validation/test dataset.
    epochs : int
        Number of training epochs.
    savepath : pathlib.Path, optional
        Directory in which to save checkpoints.
    experiment_name : str, optional
        Name prefix for saved models.
    SAVE_STATES : bool
        Whether to save model weights to disk.
    save_everyth_epoch : int
        Save checkpoints every N epochs.
    RETURN_STATES : bool
        Whether to return model checkpoints instead of only metrics.

    Returns
    -------
    dict or (dict, dict)
        If RETURN_STATES=False: returns a dict with curves.
        If True: returns (curves, saved_model_states).
    """

    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    saved_states: Dict[int, Dict[str, torch.Tensor]] = {}

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.1)
    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    for epoch in trange(epochs, desc="Training"):
        # ----------------------
        # Training
        # ----------------------
        model.train()
        running_loss = 0.0
        num_correct = 0
        total_samples = 0

        for x, y in train_data:
            optimizer.zero_grad()

            x = x.float()
            y = y.unsqueeze(1).float()

            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            preds = sigmoid(y_hat) > 0.5
            running_loss += loss.item()
            num_correct += (preds == y.bool()).sum().item()
            total_samples += y.size(0)

        train_loss[epoch] = running_loss / len(train_data)
        train_accuracy[epoch] = num_correct / total_samples

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        running_loss = 0.0
        num_correct = 0
        total_samples = 0

        with torch.no_grad():
            for x, y in test_data:
                x = x.float()
                y = y.unsqueeze(1).float()

                y_hat = model(x)
                loss = loss_fn(y_hat, y)

                preds = sigmoid(y_hat) > 0.5
                running_loss += loss.item()
                num_correct += (preds == y.bool()).sum().item()
                total_samples += y.size(0)

        test_loss[epoch] = running_loss / len(test_data)
        test_accuracy[epoch] = num_correct / total_samples

        # ----------------------
        # Save state to memory (for later analysis)
        # ----------------------
        if (epoch % save_everyth_epoch == 0) or (epoch == epochs - 1):
            saved_states[epoch] = copy.deepcopy(model.state_dict())

    # ================================================================
    # Convergence analysis
    # ================================================================
    convergence_results = determine_convergence_and_minimum(
        train_loss, test_loss, train_accuracy, test_accuracy
    )

    run_results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }

    # Save to disk if desired and training converged
    if SAVE_STATES and savepath is not None and convergence_results["converged"] and convergence_results["accurate"]:
        for epoch, state in saved_states.items():
            torch.save(state, savepath / f"epoch{epoch}.pth")

        df = pd.DataFrame.from_dict(run_results)
        (savepath / ".." / "run_summary.csv").write_text(df.to_csv(index=False))
        
        df = pd.DataFrame.from_dict([convergence_results])
        (savepath / ".." / "convergence_summary.csv").write_text(df.to_csv(index=False))
    else:
        print("Training did not converge or was not accurate; states not saved.")
        # Print accuracy and reasons
        print("Final validation accuracy:", np.mean(test_accuracy[-5:]))
        for reason in convergence_results["reasons"]:
            print(" -", reason)

    return run_results if not RETURN_STATES else (run_results, saved_states)


if __name__ == "__main__":
    pass
