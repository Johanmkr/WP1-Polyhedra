import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_blobs, make_circles
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from typing import Tuple, Dict, Callable
from torchvision import datasets, transforms

N_SAMPLES = 10000
DEFAULT_BATCH_SIZE = 32

# ------------------------------------------------------------------------------
#       1. Optimized Utility Functions
# ------------------------------------------------------------------------------

def inject_label_noise_vectorized(y: np.ndarray, noise_ratio: float, n_classes: int, seed: int) -> np.ndarray:
    """Vectorized version of label noise injection."""
    if noise_ratio <= 0.0:
        return y

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    shifts = rng.integers(low=1, high=n_classes, size=n_noisy)
    y_noisy[noisy_indices] = (y_noisy[noisy_indices] + shifts) % n_classes
    
    return y_noisy


def process_and_split(X: np.ndarray, y: np.ndarray, noise_level: float, test_size=0.2, seed=42, target_dim: int = None) -> Tuple[TensorDataset, TensorDataset]:
    """Unified pipeline for splitting, PCA scaling, and noise injection."""
    # 1. Encode labels
    unique_classes = np.sort(np.unique(y))
    class_map = {val: i for i, val in enumerate(unique_classes)}
    y_mapped = np.array([class_map[val] for val in y], dtype=np.int64)
    n_classes = len(unique_classes)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=test_size, random_state=seed, stratify=y_mapped
    )

    # 3. Inject Noise (Training labels only)
    y_train = inject_label_noise_vectorized(y_train, noise_level, n_classes, seed)

    # 4. Dimensionality Reduction (PCA) - Fit ONLY on Train Data
    if target_dim is not None:
        if target_dim > X_train.shape[1]:
            raise ValueError(f"target_dim ({target_dim}) cannot be larger than actual dataset dimension ({X_train.shape[1]}).")
        
        pca = PCA(n_components=target_dim, random_state=seed)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # 5. Scale to Unit Hypercube [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    X_test = np.clip(X_test, -1.0, 1.0)

    # 6. Tensorize
    return (
        TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64)),
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))
    )

# ------------------------------------------------------------------------------
#       2. Dataset Loaders (Lazy Loading)
# ------------------------------------------------------------------------------

def _load_uci(id: int, target_col: str = None, target_val: str = None, map_func: Callable = None):
    """Generic helper to load UCI datasets only when requested."""
    print(f"Fetching UCI dataset ID={id}...")
    dataset = fetch_ucirepo(id=id)
    X = dataset.data.features
    y = dataset.data.targets
    
    if target_col:
        y = y[target_col]
    if target_val:
        y = (y == target_val).astype(int)
    if map_func:
        X, y = map_func(X, y)
        
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int64)

def _map_car_data(X_raw, y_raw):
    mapping = {
        "buying":   {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "maint":    {"low": 0, "med": 1, "high": 2, "vhigh": 3},
        "doors":    {"2": 0, "3": 1, "4": 2, "5more": 3},
        "persons":  {"2": 0, "4": 1, "more": 2},
        "lug_boot": {"small": 0, "med": 1, "big": 2},
        "safety":   {"low": 0, "med": 1, "high": 2}
    }
    X = X_raw.copy()
    for col, counts in mapping.items():
        X[col] = X[col].map(counts)
    target_mapping = {"unacc": 0, "acc": 1, "good": 2, "vgood": 3}
    y = y_raw.iloc[:, 0].map(target_mapping)
    return X, y

# ------------------------------------------------------------------------------
#       3. The Registry (Factory Pattern)
# ------------------------------------------------------------------------------

def get_new_data(dataset_name: str, noise: float = 0.0, batch_size: int = DEFAULT_BATCH_SIZE, split_seed=42, target_dim: int = None, **kwargs):
    """
    Central entry point. Handles logic dispatch cleanly.
    target_dim: Desired number of features after applying PCA.
    """
    dataset_name = dataset_name.lower()
    
    # --- Synthetic Datasets ---
    if dataset_name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim) 
        
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim)

    elif dataset_name == "blobs":
        centers = kwargs.get("centers", 3)
        n_features = kwargs.get("n_features", 2)
        X, y = make_blobs(n_samples=N_SAMPLES, centers=centers, n_features=n_features, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed, target_dim=target_dim)

    # --- Standard Vision Datasets (Modified for Eager PCA Support) ---
    elif dataset_name in ["mnist", "mnist_minimal"]:
        print(f"Fetching {dataset_name} (torchvision)...")
        train_data = datasets.MNIST(root='./data', train=True, download=True)
        test_data = datasets.MNIST(root='./data', train=False, download=True)
        
        # Extract tensors natively to [0, 1] bounds
        X_train = train_data.data.float() / 255.0
        X_test = test_data.data.float() / 255.0
        y_train = train_data.targets.numpy()
        y_test = test_data.targets.numpy()
        
        if dataset_name == "mnist_minimal":
            X_train = torch.nn.functional.avg_pool2d(X_train.unsqueeze(1), kernel_size=4).squeeze(1)
            X_test = torch.nn.functional.avg_pool2d(X_test.unsqueeze(1), kernel_size=4).squeeze(1)
            
        # Flatten to 2D numpy arrays for sklearn compatibility
        X_train = X_train.view(X_train.size(0), -1).numpy()
        X_test = X_test.view(X_test.size(0), -1).numpy()
        
        train_ds, test_ds = process_and_split(X_train, y_train, noise_level=noise, test_size=0.2, seed=split_seed, target_dim=target_dim)
        
        # Override test set to keep the strict 60k/10k predefined splits instead of random splitting train data
        # Process test set manually to mirror the train set operations
        if noise > 0.0:
            y_train = inject_label_noise_vectorized(y_train, noise, 10, split_seed)
            
        if target_dim is not None:
             if target_dim > X_train.shape[1]:
                 raise ValueError(f"target_dim ({target_dim}) cannot be larger than actual dataset dimension ({X_train.shape[1]}).")
             pca = PCA(n_components=target_dim, random_state=split_seed)
             X_train = pca.fit_transform(X_train)
             X_test = pca.transform(X_test)
        
        # Scale back to bounded interval
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        X_test = np.clip(X_test, -1.0, 1.0)
        
        train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.int64))
        test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.int64))

    # --- UCI Datasets ---
    elif dataset_name == "wbc":
        X, y = _load_uci(id=17, target_col="Diagnosis", target_val="M")
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    elif dataset_name == "wine":
        X, y = _load_uci(id=109)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    elif dataset_name == "hd":
        X, y = _load_uci(id=45)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)

    elif dataset_name == "car":
        X, y = _load_uci(id=19, map_func=_map_car_data)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed, target_dim=target_dim)
        
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )