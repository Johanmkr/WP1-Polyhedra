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
