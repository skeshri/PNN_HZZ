import torch
import torch.nn as nn
import torch.nn.functional as F

class PNN(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        # -------------------------
        # Encoder
        # -------------------------
        self.encoder = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # -------------------------
        # Classifier (+1 for mass)
        # -------------------------
        self.classifier = nn.Sequential(
            nn.Linear(64 + 1, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.GELU(),

            nn.Linear(32, 3)
        )

    def forward(self, x, m):
        """
        x : (batch, n_features)
        m : (batch, 1) mass parameter
        returns raw logits (NO softmax)
        """
        h = self.encoder(x)
        h = torch.cat([h, m], dim=1)
        return self.classifier(h)
