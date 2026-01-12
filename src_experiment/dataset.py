import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

breast_cancer = fetch_ucirepo(id=17)

def make_wbc_dataset(noise_ratio=0.0, random_state=42):
    """
    noise_ratio: float in [0, 1]
        Fraction of training labels to corrupt with symmetric label noise
    """

    # Fetch dataset
    X_bc = breast_cancer.data.features.to_numpy(dtype="float32")
    y_bc = (breast_cancer.data.targets["Diagnosis"] == "M") \
            .astype(int).to_numpy(dtype="int64")

    # Train / test split (test labels stay clean)
    X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=random_state, stratify=y_bc
    )

    # ---- Inject symmetric label noise into training labels ----
    if noise_ratio > 0.0:
        rng = np.random.default_rng(random_state)
        n_samples = len(y_bc_train)
        n_noisy = int(noise_ratio * n_samples)

        noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

        # Binary symmetric noise: flip labels (0 ↔ 1)
        y_bc_train = y_bc_train.copy()
        y_bc_train[noisy_indices] = 1 - y_bc_train[noisy_indices]

    # Scale features
    scaler = StandardScaler()
    X_bc_train = scaler.fit_transform(X_bc_train)
    X_bc_test = scaler.transform(X_bc_test)

    # Convert to tensors
    X_bc_train = torch.tensor(X_bc_train, dtype=torch.float32)
    y_bc_train = torch.tensor(y_bc_train, dtype=torch.int64)

    X_bc_test = torch.tensor(X_bc_test, dtype=torch.float32)
    y_bc_test = torch.tensor(y_bc_test, dtype=torch.int64)

    # TensorDatasets
    bc_train_dataset = TensorDataset(X_bc_train, y_bc_train)
    bc_test_dataset = TensorDataset(X_bc_test, y_bc_test)

    return bc_train_dataset, bc_test_dataset




def make_moons_dataset(n_samples, noise, seed):
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=seed)
    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)
    return TensorDataset(X, y)



noises = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

training_seed = 0
testing_seed = 1

n_samples_training = 10_000
n_samples_testing = 5_000




def get_data(
    dataset_name: str,
    dataset_type: str = "training",
    noise: float = 0.05,
    batch_size: int = 200,
):
    if dataset_name == "moons":
        if dataset_type == "training":
            dataset = make_moons_dataset(
                n_samples=n_samples_training,
                noise=noise,
                seed=training_seed,
            )
            shuffle = True
        elif dataset_type == "testing":
            dataset = make_moons_dataset(
                n_samples=n_samples_testing,
                noise=noise,
                seed=testing_seed,
            )
            shuffle = False
        else:
            raise ValueError("dataset_type must be 'training' or 'testing'")

    elif dataset_name == "breast_cancer":
        bc_train, bc_test = make_wbc_dataset()
        if dataset_type == "training":
            dataset = bc_train
            shuffle = True
        elif dataset_type == "testing":
            dataset = bc_test
            shuffle = False
        else:
            raise ValueError("dataset_type must be 'training' or 'testing'")

    else:
        raise ValueError(f"Unknown dataset_name: {dataset_name}")

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


if __name__=="__main__":
    pass