import copy
import pathlib as pl
from typing import Dict, Tuple, List, Optional, Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
import pandas as pd
from .utils import createfolders


# ================================================================
# Training Loop
# ================================================================

def train_model_multiclass(
    model: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    num_classes: int = None,
    savepath: Optional[pl.Path] = None,
    SAVE_STATES: bool = False,
    save_everyth_epoch: int = None,
    save_for_epochs: list = None,
    RETURN_STATES: bool = False,
    sgd_lr=0.01,
    sgd_mom=0.9,
    on_save_callback: Optional[Callable[[int, Dict, Dict], None]] = None,
    disable_progress: bool = False
) -> Any:
    """
    Train a PyTorch model for multiclass classification.

    Parameters:
        model: nn.Module
            The PyTorch model to train. Output layer should have `num_classes` units.
        train_data: DataLoader
            Training data.
        test_data: DataLoader
            Validation/test data.
        epochs: int
            Number of training epochs.
        num_classes: int
            Number of classes.
        savepath: Optional[pathlib.Path]
            Path to save run summary and state_dicts (if using legacy disk saving).
        SAVE_STATES: bool
            Whether to save model states to disk internally (legacy behavior).
        save_everyth_epoch: int
            Interval of epochs to save state (e.g., every 50 epochs).
        save_for_epochs: list
            Specific list of epochs to save (e.g., [0, 1, 2, 5, 10]).
        RETURN_STATES: bool
            Whether to return saved model states in memory.
        sgd_lr: float
            Learning rate for SGD.
        sgd_mom: float
            Momentum for SGD.
        on_save_callback: Callable[[int, Dict, Dict], None]
            Optional callback function(epoch, state_dict, metrics) to handle incremental saving 
            (e.g., writing to HDF5 during training).
    
    Returns:
        Tuple of (results_df,) or (results_df, saved_states)
    """

    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)
    eval_train_loss = np.zeros(epochs)
    eval_train_accuracy = np.zeros(epochs)
    
    if num_classes is None:
        num_classes = model.num_classes
    
    saved_states: Dict[int, Dict[str, torch.Tensor]] = {}

    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_mom)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in trange(epochs, desc="Training", leave=False, disable=disable_progress):
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
            y = y.long()  # labels as integers (0,1,...,num_classes-1)

            y_hat = model(x)  # raw logits, shape (batch_size, num_classes)
            loss = loss_fn(y_hat, y)
            loss.backward()
            optimizer.step()

            preds = y_hat.argmax(dim=1)
            
            running_loss += loss.item()
            num_correct += (preds == y).sum().item()
            total_samples += y.size(0)

        train_loss[epoch] = running_loss / len(train_data)
        train_accuracy[epoch] = num_correct / total_samples

        # ----------------------
        # Validation and evaluation
        # ----------------------
        model.eval()
        
        # Eval on train data (for analysis consistency)
        train_running_loss = 0.0
        train_num_correct = 0
        train_total_samples = 0
        with torch.no_grad():
            for x, y in train_data:
                x = x.float()
                y = y.long()
                
                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                preds = y_hat.argmax(dim=1)
                
                train_running_loss += loss.item()
                train_num_correct += (preds == y).sum().item()
                train_total_samples += y.size(0)
        eval_train_loss[epoch] = train_running_loss / len(train_data)
        eval_train_accuracy[epoch] = train_num_correct / train_total_samples
            
        # Eval on test data
        test_running_loss = 0.0
        test_num_correct = 0
        test_total_samples = 0
        with torch.no_grad():
            for x, y in test_data:
                x = x.float()
                y = y.long()

                y_hat = model(x)
                loss = loss_fn(y_hat, y)
                preds = y_hat.argmax(dim=1)
                
                test_running_loss += loss.item()
                test_num_correct += (preds == y).sum().item()
                test_total_samples += y.size(0)

        test_loss[epoch] = test_running_loss / len(test_data)
        test_accuracy[epoch] = test_num_correct / test_total_samples

        # ----------------------
        # Save state Logic
        # ----------------------
        should_save = False

        # Condition A: Is it in the specific list? (e.g. [0, 1, 2, 5])
        if save_for_epochs is not None and epoch in save_for_epochs:
            should_save = True
        
        # Condition B: Does it match the interval? (e.g. every 50)
        if save_everyth_epoch is not None:
            if (epoch % save_everyth_epoch == 0) or (epoch == epochs - 1):
                should_save = True

        if should_save:
            state = copy.deepcopy(model.state_dict())
            
            # Use Callback if provided (Best Practice for incremental HDF5 saving)
            if on_save_callback:
                metrics = {
                    "train_loss": train_loss[epoch],
                    "train_accuracy": train_accuracy[epoch],
                    "test_loss": test_loss[epoch],
                    "test_accuracy": test_accuracy[epoch],
                    "eval_train_loss": eval_train_loss,
                    "eval_train_accuracy": eval_train_accuracy,
                }
                on_save_callback(epoch, state, metrics)
            
            # Fallback to dictionary (Legacy in-memory or disk behavior)
            elif RETURN_STATES or SAVE_STATES:
                saved_states[epoch] = state

    # ----------------------
    # Compile results
    # ----------------------
    run_results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "eval_train_loss": eval_train_loss,
        "eval_train_accuracy": eval_train_accuracy,
    }
    results = pd.DataFrame.from_dict(run_results)

    # Legacy: Save to disk as .pth files if requested and NO callback is used
    if SAVE_STATES and savepath is not None and not on_save_callback:
        createfolders(savepath / "state_dicts")
        for epoch, state in saved_states.items():
            torch.save(state, savepath / "state_dicts" / f"epoch{epoch}.pth")
        (savepath / "run_summary.csv").write_text(results.to_csv(index=False))
    elif not on_save_callback and not RETURN_STATES:
        print("Training not saved.")

    return (results,) if not RETURN_STATES else (results, saved_states)


if __name__ == "__main__":
    pass