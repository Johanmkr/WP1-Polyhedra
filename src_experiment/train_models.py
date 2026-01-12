import copy
import pathlib as pl
from typing import Dict, Tuple, List, Optional, Any

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

def train_model(
    model: nn.Module,
    train_data: DataLoader,
    test_data: DataLoader,
    epochs: int,
    savepath: Optional[pl.Path] = None,
    SAVE_STATES: bool = False,
    save_everyth_epoch: int = 50,
    RETURN_STATES: bool = False,
    sgd_lr = 0.01,
    sgd_mom = 0.9
) -> Any:

    train_loss = np.zeros(epochs)
    train_accuracy = np.zeros(epochs)
    test_loss = np.zeros(epochs)
    test_accuracy = np.zeros(epochs)

    saved_states: Dict[int, Dict[str, torch.Tensor]] = {}

    optimizer = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=sgd_mom)
    loss_fn = nn.BCEWithLogitsLoss()
    sigmoid = nn.Sigmoid()

    for epoch in trange(epochs, desc="Training", leave=False):
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
        if (epoch % save_everyth_epoch == 0) or (epoch == epochs-1):
            saved_states[epoch] = copy.deepcopy(model.state_dict())


    run_results = {
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    results = pd.DataFrame.from_dict(run_results)

    # Save to disk if desired and training converged
    if SAVE_STATES and savepath is not None:
        createfolders(savepath / "state_dicts")
        for epoch, state in saved_states.items():
            torch.save(state, savepath / "state_dicts" / f"epoch{epoch}.pth")
        (savepath / "run_summary.csv").write_text(results.to_csv(index=False))
        
    else:
        print("Training not saved.")


    return (results,) if not RETURN_STATES else (results, saved_states)


if __name__ == "__main__":
    pass
