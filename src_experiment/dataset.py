import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

N_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 32


# ------------------------------------------------------------------------------
#       Utility funcs
# ------------------------------------------------------------------------------


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

import numpy as np

def inject_symmetric_multivariate_noise(
        y: np.ndarray,
        noise_ratio: float,
        n_classes: int,
        seed: int,
    ) -> np.ndarray:
        """
        Apply symmetric label noise to multi-class labels.
        """
        if noise_ratio <= 0.0:
            return y

        rng = np.random.default_rng(seed)
        y_noisy = y.copy().flatten() # Ensure it's 1D

        n_samples = len(y_noisy)
        n_noisy = int(noise_ratio * n_samples)
        
        # Pick which indices to corrupt
        noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)

        for idx in noisy_indices:
            current_label = y_noisy[idx]
            
            # Create list of possible alternative labels (excluding the correct one)
            possible_labels = [i for i in range(n_classes) if i != current_label]
            
            # Randomly select one of the other labels
            y_noisy[idx] = rng.choice(possible_labels)

        return y_noisy

def to_tensor_dataset(X: np.ndarray, y: np.ndarray) -> TensorDataset:
    return TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64),
    )
    
def split_and_scale(X, y, noise, test_size=0.2, state=42):
    # Map labels to 0..num_classes-1
    unique_classes = np.sort(np.unique(y))
    class_map = {int(c): i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[int(val)] for val in y], dtype=np.int64)


    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=state,
        stratify=y,
    )

    # Inject label noise into training labels only
    y_train = inject_symmetric_multivariate_noise(y_train, noise, len(unique_classes), state)

    # Standardize features (fit on train only)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return (
        to_tensor_dataset(X_train, y_train),
        to_tensor_dataset(X_test, y_test),
    )
    

def split_and_scale_moons(
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


# ------------------------------------------------------------------------------
#       Datasets
# ------------------------------------------------------------------------------

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
    
    # Map labels to 0..num_classes-1
    unique_classes = np.sort(np.unique(y))
    class_map = {int(c): i for i, c in enumerate(unique_classes)}
    y = np.array([class_map[int(val)] for val in y], dtype=np.int64)

    X_train, X_test, y_train, y_test = split_and_scale_moons(
        X,
        y,
        test_size=0.2,
        random_state=random_state,
    )

    return (
        to_tensor_dataset(X_train, y_train),
        to_tensor_dataset(X_test, y_test),
    )

breast_cancer = fetch_ucirepo(id=17)
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
    
    return split_and_scale(X, y, label_noise)

    # # Map labels to 0..num_classes-1
    # unique_classes = np.sort(np.unique(y))
    # class_map = {int(c): i for i, c in enumerate(unique_classes)}
    # y = np.array([class_map[int(val)] for val in y], dtype=np.int64)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     random_state=random_state,
    #     stratify=y,
    # )

    # # Inject label noise into training labels only
    # y_train = inject_symmetric_label_noise(
    #     y_train,
    #     noise_ratio=label_noise,
    #     seed=random_state,
    # )

    # # Standardize features (fit on train only)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # return (
    #     to_tensor_dataset(X_train, y_train),
    #     to_tensor_dataset(X_test, y_test),
    # )

# wine_quality = fetch_ucirepo(id=186)
wine = fetch_ucirepo(id=109) 
def make_wine_data(
    label_noise: float,
    random_state: int = 42
) -> tuple[TensorDataset, TensorDataset]:
    # X = wine_quality.data.features.to_numpy(dtype=np.float32)
    # y = wine_quality.data.targets.to_numpy(dtype=np.int64)
    X = wine.data.features.to_numpy(dtype=np.float32) 
    y = wine.data.targets.to_numpy(dtype=np.int64)

    return split_and_scale(X, y, label_noise)

    # # Map labels to 0..num_classes-1
    # unique_classes = np.sort(np.unique(y))
    # class_map = {int(c): i for i, c in enumerate(unique_classes)}
    # y = np.array([class_map[int(val)] for val in y], dtype=np.int64)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     random_state=random_state,
    #     stratify=y,
    # )

    # # Inject label noise into training labels only
    # y_train = inject_symmetric_multivariate_noise(y_train, label_noise, len(unique_classes), random_state)

    # # Standardize features (fit on train only)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # return (
    #     to_tensor_dataset(X_train, y_train),
    #     to_tensor_dataset(X_test, y_test),
    # )


heart_disease = fetch_ucirepo(id=45) 
def make_hd_data(
    label_noise: float,
    random_state: int = 42
) -> tuple[TensorDataset, TensorDataset]:
    X = heart_disease.data.features.to_numpy(dtype=np.float32)
    y = heart_disease.data.targets.to_numpy(dtype=np.int64)

    X[np.isnan(X)] = 0
    y[np.isnan(y)] = 0
    
    return split_and_scale(X, y, label_noise)
    
    # # Map labels to 0..num_classes-1
    # unique_classes = np.sort(np.unique(y))
    # class_map = {int(c): i for i, c in enumerate(unique_classes)}
    # y = np.array([class_map[int(val)] for val in y], dtype=np.int64)

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     random_state=random_state,
    #     stratify=y,
    # )

    # # Inject label noise into training labels only
    # y_train = inject_symmetric_multivariate_noise(y_train, label_noise, len(unique_classes), random_state)

    # # Standardize features (fit on train only)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # return (
    #     to_tensor_dataset(X_train, y_train),
    #     to_tensor_dataset(X_test, y_test),
    # )



car_evaluation = fetch_ucirepo(id=19) 
def make_car_data(
    label_noise: float,
    random_state: int = 42
) -> tuple[TensorDataset, TensorDataset]:
    
    X_raw = car_evaluation.data.features
    y_raw = car_evaluation.data.targets
    # Define the order for each feature
    mapping = {
        "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
        "persons":  {"2": 0, "4": 1, "more": 2},
        "lug_boot": {"small": 0, "med": 1, "big": 2},
        "safety":   {"low": 0, "med": 1, "high": 2}
    }
    # Apply mapping to X
    X_encoded = X_raw.copy()
    for col, counts in mapping.items():
        X_encoded[col] = X_encoded[col].map(counts)
        
    # Map the target y (unacc, acc, good, vgood)
    target_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
    y_encoded = y_raw.iloc[:, 0].map(target_mapping) # Ensure y is a Series

    X = X_encoded.to_numpy(dtype=np.float32)
    y = y_encoded.to_numpy(dtype=np.int64)
    
    return split_and_scale(X, y, label_noise)
    
    # # Map labels to 0..num_classes-1
    # unique_classes = np.sort(np.unique(y))
    # class_map = {int(c): i for i, c in enumerate(unique_classes)}
    # y = np.array([class_map[int(val)] for val in y], dtype=np.int64)


    # X_train, X_test, y_train, y_test = train_test_split(
    #     X,
    #     y,
    #     test_size=0.2,
    #     random_state=random_state,
    #     stratify=y,
    # )

    # # Inject label noise into training labels only
    # y_train = inject_symmetric_multivariate_noise(y_train, label_noise, len(unique_classes), random_state)

    # # Standardize features (fit on train only)
    # scaler = StandardScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)

    # return (
    #     to_tensor_dataset(X_train, y_train),
    #     to_tensor_dataset(X_test, y_test),
    # )








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


# def get_wbc_data(
#     label_noise: float = 0.0,
#     batch_size: int = DEFAULT_BATCH_SIZE,
# ) -> DataLoader:
#     """
#     Return a DataLoader for the Wisconsin Breast Cancer dataset.

#     dataset_type: "training" or "testing"
#     label_noise: symmetric label noise applied to training labels only
#     """
    
#     train_ds, test_ds = make_wbc_datasets(label_noise=label_noise)


#     return DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#     ), DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#     )
    
 
# def get_wine_data(
#     label_noise: float = 0.0,
#     batch_size = DEFAULT_BATCH_SIZE,
# ) -> DataLoader:
    
#     train_ds, test_ds = make_wine_data(label_noise=label_noise)
    
#     return DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#     ), DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#     )
    
# def get_hd_data(
#     label_noise: float = 0.0,
#     batch_size = DEFAULT_BATCH_SIZE,
# ) -> DataLoader:
    
#     train_ds, test_ds = make_hd_data(label_noise=label_noise)
    
#     return DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#     ), DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#     )
    
# def get_car_data(
#     label_noise: float = 0.0,
#     batch_size = DEFAULT_BATCH_SIZE,
# ) -> DataLoader:
    
#     train_ds, test_ds = make_car_data(label_noise=label_noise)
    
#     return DataLoader(
#         train_ds,
#         batch_size=batch_size,
#         shuffle=True,
#     ), DataLoader(
#         test_ds,
#         batch_size=batch_size,
#         shuffle=False,
#     )
    
def get_new_data(dataset, noise=0, batch_size=DEFAULT_BATCH_SIZE):
    assert dataset in ["moons", "wbc", "wine", "hd", "car"], "Invalid dataset"
    match dataset:
        case "moons":
            train_ds, test_ds = make_moons_datasets(feature_noise=noise)
        case "wbc":
            train_ds, test_ds = make_wbc_datasets(label_noise=noise)
        case "wine":
            train_ds, test_ds = make_wine_data(label_noise=noise)
        case "hd":
            train_ds, test_ds = make_hd_data(label_noise=noise)
        case "car":
            train_ds, test_ds = make_car_data(label_noise=noise)
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
