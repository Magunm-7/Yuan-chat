"""
Multimodal, temporal MPSE (the locked architecture, notes.md 6.8/6.9).

  per-modality embeddings --proj--> H
  sigmoid alpha gate over modalities (independent 0..1, NOT softmax; A7)
  weighted fusion --> GRU over the session's turns (temporal; A5/F2/F5)
  heads: mu (sigmoid), logvar -> sigma (heteroscedastic)

Batch = one session (variable-length turn sequence); simplest and fast enough for
128 sessions. `modalities` selects the active streams, so the text-only baseline
is the same model with modalities=("text",).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class MPSE_MM(nn.Module):
    def __init__(self, dims: dict[str, int], modalities: tuple[str, ...],
                 hidden: int = 256, num_idx: int = 1, gru_layers: int = 1, dropout: float = 0.1,
                 use_gru: bool = True, use_sigmoid: bool = True, use_alpha: bool = True):
        super().__init__()
        self.mods = tuple(modalities)
        self.use_gru = use_gru
        self.use_sigmoid = use_sigmoid
        self.use_alpha = use_alpha
        self.proj = nn.ModuleDict({
            m: nn.Sequential(nn.Linear(dims[m], hidden), nn.ReLU(), nn.Dropout(dropout))
            for m in self.mods
        })
        # sigmoid gate per modality, computed from concatenated modality reps
        self.alpha_head = nn.Linear(hidden * len(self.mods), len(self.mods))
        self.gru = nn.GRU(hidden, hidden, num_layers=gru_layers, batch_first=True) if use_gru else None
        self.mu_head = nn.Linear(hidden, num_idx)
        self.logvar_head = nn.Linear(hidden, num_idx)

    def forward(self, feats: dict):
        """feats[m]: (B, T, dims[m]).  Returns mu, sigma, alpha, logvar."""
        reps = [self.proj[m](feats[m]) for m in self.mods]      # each (B,T,H)
        stack = torch.stack(reps, dim=2)                        # (B,T,M,H)
        if self.use_alpha:
            cat = torch.cat(reps, dim=-1)                       # (B,T,M*H)
            alpha = torch.sigmoid(self.alpha_head(cat))         # (B,T,M) independent gates
        else:  # ablation: uniform (equal-weight) fusion, no learned gate
            B, T, M = stack.shape[0], stack.shape[1], stack.shape[2]
            alpha = torch.full((B, T, M), 1.0 / M, device=stack.device, dtype=stack.dtype)
        fused = (stack * alpha.unsqueeze(-1)).sum(dim=2)        # (B,T,H)
        out = self.gru(fused)[0] if self.use_gru else fused    # (B,T,H)
        raw_mu = self.mu_head(out)
        mu = torch.sigmoid(raw_mu) if self.use_sigmoid else raw_mu   # (B,T,num_idx)
        logvar = torch.clamp(self.logvar_head(out), -6.0, 2.0)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma, alpha, logvar


def hetero_nll(mu, logvar, y, weight=None):
    """Gaussian heteroscedastic NLL, optionally per-turn weighted. Shapes (B,T,K)."""
    var = torch.exp(logvar)
    nll = 0.5 * ((y - mu) ** 2 / (var + 1e-8) + logvar)        # (B,T,K)
    nll = nll.mean(dim=-1)                                      # (B,T)
    if weight is not None:
        nll = nll * weight                                     # (B,T)
    return nll.mean()
