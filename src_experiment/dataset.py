import torch
from sklearn.datasets import make_moons
# 2D - classification dataet

class Classification(torch.utils.data.Dataset):
    def __init__(self, x, y, device="cpu"):
        self.X = torch.from_numpy(x).to(device)
        self.Y = torch.from_numpy(y).to(device)

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        return (self.X[idx], self.Y[idx])


def make_moon_dataloader(n_samples=1000, noise=0.1, random_state=42, batch_size=100):
    # Make data
    dataset = Classification(*make_moons(n_samples=n_samples, noise=noise, random_state=random_state))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


# Hard coded datasets for convenience

# General paramters
noises = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
training_seed = 0
testing_seed = 1
inference_seed = 2
batch_size = 200

# Small moons dataset parameters
small_moon_training_params = {
    "n_samples": 1000,
    "random_state": training_seed,
    "batch_size": batch_size,
}

small_moon_testing_params = {
    "n_samples": 200,
    "random_state": testing_seed,
    "batch_size": batch_size,
}
small_moon_inference_params = {
    "n_samples": 200,
    "random_state": inference_seed,
    "batch_size": batch_size,
}

# Medium moons dataset parameters
medium_moon_training_params = {
    "n_samples": 5000,
    "random_state": training_seed,
    "batch_size": batch_size
}
medium_moon_testing_params = {
    "n_samples": 1000,
    "random_state": testing_seed,
    "batch_size": batch_size,
}
medium_moon_inference_params = {
    "n_samples": 1000,
    "random_state": inference_seed,
    "batch_size": batch_size,
}

# Large moons dataset parameters
large_moon_training_params = {
    "n_samples": 25000,
    "random_state": training_seed,
    "batch_size": batch_size,
}
large_moon_testing_params = {
    "n_samples": 5000,
    "random_state": testing_seed,
    "batch_size": batch_size,
}
large_moon_inference_params = {
    "n_samples": 5000,
    "random_state": inference_seed,
    "batch_size": batch_size,
}

def get_moon_dataloaders(size="small", noise=0.1):
    if size=="small":
        train_params = small_moon_training_params
        test_params = small_moon_testing_params
        inference_params = small_moon_inference_params
    elif size=="medium":
        train_params = medium_moon_training_params
        test_params = medium_moon_testing_params
        inference_params = medium_moon_inference_params
    elif size=="large":
        train_params = large_moon_training_params
        test_params = large_moon_testing_params
        inference_params = large_moon_inference_params
    else:
        raise ValueError("Size must be one of 'small', 'medium', or 'large'.")

    train_loader = make_moon_dataloader(noise=noise, **train_params)
    test_loader = make_moon_dataloader(noise=noise, **test_params)
    inference_loader = make_moon_dataloader(noise=noise, **inference_params)

    return train_loader, test_loader, inference_loader

datasets = {}
for size in ["small","medium","large"]:
    datasets[size] = {}
    for noise in noises:
        train_loader, test_loader, inference_loader = get_moon_dataloaders(size=size, noise=noise)
        datasets[size][noise] = {
            "train": train_loader,
            "test": test_loader,
            "inference": inference_loader
        }

if __name__=="__main__":
    pass