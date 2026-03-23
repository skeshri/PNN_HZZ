import torch
import numpy as np

class PNNWrapper(torch.nn.Module):
    def __init__(self, model, mass_norm, device):
        super().__init__()
        self.model = model
        self.mass_norm = mass_norm
        self.device = device

    def forward(self, x):
        m = torch.full(
            (x.shape[0], 1),
            self.mass_norm,
            device=self.device
        )
        out = self.model(x, m)
        return out

