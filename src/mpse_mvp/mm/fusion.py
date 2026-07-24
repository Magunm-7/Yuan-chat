"""
Text-queried cross-attention fusion (route 2, replaces the pooled SoftTokenProjector).

Motivation (notes 0.4): the pooled 1-vector prefix used the audio/video too weakly
(diagnostic: only ~2/6 turns responded to the av prefix). Here the text embedding
conditions K query tokens that cross-attend the audio + video SEQUENCES, so "how
the client said it" enters generation frame-by-frame.

Two layers of evidence weighting (keeps the project's alpha idea, adds frame-level):
  - alpha gates the two modality memories        (modality-level, kept from before)
  - cross-attention selects within each sequence (frame-level, new)

Fusion runs in a small d_fuse (default 512) for parameter economy on small data
(23 low-quality sessions), then projects K tokens up to the LLM hidden dim.
"""
from __future__ import annotations
import torch
import torch.nn as nn


class CrossAttnFusion(nn.Module):
    def __init__(self, d_model: int, text_dim: int = 384, audio_dim: int = 768,
                 video_dim: int = 768, k_tokens: int = 8, d_fuse: int = 512,
                 n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.k = k_tokens
        self.d_fuse = d_fuse
        self.q_from_text = nn.Linear(text_dim, k_tokens * d_fuse)   # text conditions the K queries
        self.audio_in = nn.Linear(audio_dim, d_fuse)
        self.video_in = nn.Linear(video_dim, d_fuse)
        self.attn = nn.MultiheadAttention(d_fuse, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_fuse)
        self.ffn = nn.Sequential(
            nn.Linear(d_fuse, d_fuse * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_fuse * 2, d_fuse),
        )
        self.ln2 = nn.LayerNorm(d_fuse)
        self.out = nn.Linear(d_fuse, d_model)

    def forward(self, text_emb, audio_seq, video_seq, alpha=None):
        """
        text_emb : (B, text_dim)
        audio_seq: (B, Ta, audio_dim)   video_seq: (B, Tv, video_dim)
        alpha    : (B, 2) modality gates [audio, video], or None
        returns  : (B, K, d_model) soft-token prefix
        """
        B = text_emb.shape[0]
        q = self.q_from_text(text_emb).view(B, self.k, self.d_fuse)   # (B,K,d)
        a = self.audio_in(audio_seq)                                  # (B,Ta,d)
        v = self.video_in(video_seq)                                  # (B,Tv,d)
        if alpha is not None:
            a = a * alpha[:, 0].view(B, 1, 1)
            v = v * alpha[:, 1].view(B, 1, 1)
        mem = torch.cat([a, v], dim=1)                                # (B,Ta+Tv,d)
        att, _ = self.attn(q, mem, mem)                              # (B,K,d)
        h = self.ln1(q + att)
        h = self.ln2(h + self.ffn(h))
        return self.out(h)                                           # (B,K,d_model)
