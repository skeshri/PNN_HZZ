import torch
from torch.utils.data import Dataset
import numpy as np

class PNN_Dataset(Dataset):
    def __init__(self, npz_file, norm_file=None):
        data = np.load(npz_file)

        X = data["X"]
        y = data["y"]
        m = data["m"]

        # --- Normalize features ---
        if norm_file is None:
            self.mean = X.mean(axis=0)
            self.std  = X.std(axis=0) + 1e-6
        else:
            norm = np.load(norm_file)
            self.mean = norm["mean"]
            self.std  = norm["std"]

        X = (X - self.mean) / self.std

        # --- Normalize mass ---
        m = (m - 1000.0) / 500.0

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.m = torch.tensor(m, dtype=torch.float32).unsqueeze(1)

    def save_norm(self, filename):
        np.savez(filename, mean=self.mean, std=self.std)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.m[idx], self.y[idx]

