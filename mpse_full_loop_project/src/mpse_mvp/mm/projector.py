
from __future__ import annotations
import torch
import torch.nn as nn

class SoftTokenProjector(nn.Module):
    """
    Map a pooled embedding (B, C_in) into K soft tokens in LLM hidden dim D.
    """
    def __init__(self, c_in: int, d_model: int, k_tokens: int = 8, hidden: int = 512, dropout: float = 0.0):
        super().__init__()
        self.k = k_tokens
        self.net = nn.Sequential(
            nn.Linear(c_in, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, k_tokens * d_model),
        )
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C_in) -> (B, K, D)
        B = x.shape[0]
        y = self.net(x).view(B, self.k, self.d_model)
        return y
