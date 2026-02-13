import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons, make_blobs, make_circles, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from typing import Tuple, Dict, Callable

N_SAMPLES = 1000
DEFAULT_BATCH_SIZE = 32

# ------------------------------------------------------------------------------
#       1. Optimized Utility Functions
# ------------------------------------------------------------------------------

def inject_label_noise_vectorized(y: np.ndarray, noise_ratio: float, n_classes: int, seed: int) -> np.ndarray:
    """
    Vectorized version of label noise injection. Roughly 50x faster.
    """
    if noise_ratio <= 0.0:
        return y

    rng = np.random.default_rng(seed)
    y_noisy = y.copy()
    n_samples = len(y)
    n_noisy = int(noise_ratio * n_samples)
    
    # 1. Select indices to corrupt
    noisy_indices = rng.choice(n_samples, size=n_noisy, replace=False)
    
    # 2. Vectorized shift: add a random int [1, n_classes-1] modulo n_classes
    # This guarantees the new label is different from the old one.
    shifts = rng.integers(low=1, high=n_classes, size=n_noisy)
    y_noisy[noisy_indices] = (y_noisy[noisy_indices] + shifts) % n_classes
    
    return y_noisy


def process_and_split(X: np.ndarray, y: np.ndarray, noise_level: float, test_size=0.2, seed=42) -> Tuple[TensorDataset, TensorDataset]:
    """
    Unified pipeline for splitting, scaling, and noise injection.
    """
    # 1. Encode labels to 0..K-1
    # Ensures labels are integers 0, 1, ..., K-1 regardless of input format (strings, 1-based, etc.)
    unique_classes = np.sort(np.unique(y))
    class_map = {val: i for i, val in enumerate(unique_classes)}
    # Handle mixed types if necessary, but generally assuming y is consistent
    y_mapped = np.array([class_map[val] for val in y], dtype=np.int64)
    n_classes = len(unique_classes)

    # 2. Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=test_size, random_state=seed, stratify=y_mapped
    )

    # 3. Inject Noise (Training labels only)
    # Note: For synthetic data (Moons), 'noise_level' usually refers to feature noise, 
    # handled at generation time. If you want label noise for them, uncomment below.
    y_train = inject_label_noise_vectorized(y_train, noise_level, n_classes, seed)

    # 4. Scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Tensorize
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
    
    # Handle specific target column selection
    if target_col:
        y = y[target_col]
    
    # Handle binarization (e.g., WBC Diagnosis == 'M')
    if target_val:
        y = (y == target_val).astype(int)
        
    # Handle custom mapping (e.g., Car Evaluation)
    if map_func:
        X, y = map_func(X, y)
        
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int64)

# Custom mapping for Car Evaluation
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

def get_new_data(dataset_name: str, noise: float = 0.0, batch_size: int = DEFAULT_BATCH_SIZE, split_seed=42, **kwargs):
    """
    Central entry point. Handles logic dispatch cleanly.
    """
    dataset_name = dataset_name.lower()
    
    # --- Synthetic Datasets (Feature Noise injection handled here) ---
    if dataset_name == "moons":
        X, y = make_moons(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        # Note: We pass noise=0.0 to process_and_split because we already applied feature noise
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed) 
        
    elif dataset_name == "circles":
        X, y = make_circles(n_samples=N_SAMPLES, noise=noise, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed)

    elif dataset_name == "blobs":
        centers = kwargs.get("centers", 3)
        n_features = kwargs.get("n_features", 2)
        X, y = make_blobs(n_samples=N_SAMPLES, centers=centers, n_features=n_features, random_state=split_seed)
        train_ds, test_ds = process_and_split(X, y, noise_level=0.0, seed=split_seed)

    # --- Standard Vision Datasets ---
    elif dataset_name == "mnist":
        # Fetches 70,000 samples, 784 features.
        # Note: This loads the entire dataset into RAM.
        print("Fetching MNIST (OpenML)...")
        X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed)

    # --- UCI Datasets (Label Noise injection handled in process_and_split) ---
    elif dataset_name == "wbc":
        X, y = _load_uci(id=17, target_col="Diagnosis", target_val="M")
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed)
        
    elif dataset_name == "wine":
        X, y = _load_uci(id=109)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed)
        
    elif dataset_name == "hd":
        X, y = _load_uci(id=45)
        X[np.isnan(X)] = 0
        y[np.isnan(y)] = 0
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed)

    elif dataset_name == "car":
        X, y = _load_uci(id=19, map_func=_map_car_data)
        train_ds, test_ds = process_and_split(X, y, noise_level=noise, seed=split_seed)
        
    else:
        raise ValueError(f"Invalid dataset: {dataset_name}")

    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    )