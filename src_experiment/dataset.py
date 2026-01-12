import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

N_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 200

breast_cancer = fetch_ucirepo(id=17)

def inject_symmetric_label_noise(
    y: np.ndarray,
    noise_ratio: float,
    seed: int,
) -> np.ndarray:
    """
    Apply symmetric label noise to binary labels.

    noise_ratio: fraction of labels to flip
    """
    if noise_ratio <= 0.0:
        return y

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()

    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

    y_noisy[noisy_indices] = 1 - y_noisy[noisy_indices]
    return y_noisy


def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64),
    )
    

def split_and_scale_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train/test sets and standardize features.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def make_moons_datasets(
    feature_noise: float,
    random_state: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """
    Create train/test moons datasets.

    feature_noise controls geometric noise in the input space.
    """
    X, y = make_moons(
        n_samples=N_SAMPLES,
        noise=feature_noise,
        random_state=random_state,
    )

    X_train, X_test, y_train, y_test = split_and_scale_data(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    return (
        to_tensor_dataset(X_train, y_train),
        to_tensor_dataset(X_test, y_test),
    )


def make_wbc_datasets(
    label_noise: float,
    random_state: int = 42,
) -> tuple[TensorDataset, TensorDataset]:
    """
    Create train/test datasets for the Wisconsin Breast Cancer dataset.

    label_noise is symmetric label corruption applied ONLY to training labels.
    """

    X = breast_cancer.data.features.to_numpy(dtype=np.float32)
    y = (
        breast_cancer.data.targets["Diagnosis"] == "M"
    ).astype(np.int64).to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
        stratify=y,
    )

    # Inject label noise into training labels only
    y_train = inject_symmetric_label_noise(
        y_train,
        noise_ratio=label_noise,
        seed=random_state,
    )

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        to_tensor_dataset(X_train, y_train),
        to_tensor_dataset(X_test, y_test),
    )

def get_moons_data(
    feature_noise: float = 0.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DataLoader:
    """
    Return a DataLoader for the moons dataset.

    dataset_type: "training" or "testing"
    feature_noise: geometric noise level in the input space
    """
    train_ds, test_ds = make_moons_datasets(feature_noise=feature_noise)

    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    ), DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )


def get_wbc_data(
    label_noise: float = 0.0,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> DataLoader:
    """
    Return a DataLoader for the Wisconsin Breast Cancer dataset.

    dataset_type: "training" or "testing"
    label_noise: symmetric label noise applied to training labels only
    """
    
    train_ds, test_ds = make_wbc_datasets(label_noise=label_noise)


    return DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
    ), DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
    )


if __name__ == "__main__":
    pass
