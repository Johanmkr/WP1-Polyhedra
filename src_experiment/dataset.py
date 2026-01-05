import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo


def make_wbc_dataset():
    # Fetch dataset
    breast_cancer = fetch_ucirepo(id=17)

    X_bc = breast_cancer.data.features.to_numpy(dtype="float32")
    y_bc = (breast_cancer.data.targets["Diagnosis"] == "M").astype(int).to_numpy(dtype="int64")

    # Train / test split
    X_bc_train, X_bc_test, y_bc_train, y_bc_test = train_test_split(
        X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
    )

    # Scale features
    scaler = StandardScaler()
    X_bc_train = scaler.fit_transform(X_bc_train)
    X_bc_test = scaler.transform(X_bc_test)

    # Convert to tensors
    X_bc_train = torch.tensor(X_bc_train)
    y_bc_train = torch.tensor(y_bc_train)

    X_bc_test = torch.tensor(X_bc_test)
    y_bc_test = torch.tensor(y_bc_test)

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