from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MPSE(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_indices: int, dropout: float = 0.1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # gating alpha over 3 modalities
        self.alpha_head = nn.Linear(hidden_dim, 3)

        # per-index heads: mu and logvar
        self.mu_head = nn.Linear(hidden_dim, num_indices)
        self.logvar_head = nn.Linear(hidden_dim, num_indices)

    def forward(self, x):
        h = self.fc(x)
        alpha = F.softmax(self.alpha_head(h), dim=-1)
        mu = torch.sigmoid(self.mu_head(h))  # [0,1]
        logvar = torch.clamp(self.logvar_head(h), -6.0, 2.0)
        sigma = torch.exp(0.5*logvar)
        return mu, sigma, alpha, logvar
