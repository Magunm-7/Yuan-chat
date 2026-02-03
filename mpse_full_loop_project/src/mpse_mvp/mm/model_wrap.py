
from __future__ import annotations
import torch
import torch.nn as nn

from .projector import SoftTokenProjector

class MultiModalPrefixLM(nn.Module):
    """
    LLaMA + (audio/video)->prefix soft tokens.
    Freeze base LM by default; train projectors (and optional small head).
    """
    def __init__(self, base_lm, d_model: int, audio_c: int, video_c: int,
                 k_audio: int = 8, k_video: int = 8, projector_hidden: int = 512,
                 train_base: bool = False, use_alpha_gate: bool = True,
                 aux_mu_dim: int = 0):
        super().__init__()
        self.lm = base_lm
        self.d_model = d_model
        self.use_alpha_gate = use_alpha_gate

        self.audio_proj = SoftTokenProjector(audio_c, d_model, k_tokens=k_audio, hidden=projector_hidden)
        self.video_proj = SoftTokenProjector(video_c, d_model, k_tokens=k_video, hidden=projector_hidden)

        # optional auxiliary head to predict mu (state indices)
        self.aux_mu_dim = aux_mu_dim
        if aux_mu_dim > 0:
            self.mu_head = nn.Linear(d_model, aux_mu_dim)
        else:
            self.mu_head = None

        if not train_base:
            for p in self.lm.parameters():
                p.requires_grad_(False)

    def forward(self, input_ids, attention_mask, labels,
                audio_feat, video_feat, alpha=None, sample_weight=None, mu_target=None):
        """
        audio_feat: (B, Ca) pooled
        video_feat: (B, Cv) pooled
        alpha: (B, 2) or dict with audio/video weights
        """
        B = input_ids.shape[0]
        # token embeddings
        emb = self.lm.get_input_embeddings()(input_ids)

        # prefix
        a_tok = self.audio_proj(audio_feat)  # (B,Ka,D)
        v_tok = self.video_proj(video_feat)  # (B,Kv,D)

        if self.use_alpha_gate and alpha is not None:
            # allow alpha as dict or tensor
            if isinstance(alpha, dict):
                a_w = float(alpha.get("audio", 0.5))
                v_w = float(alpha.get("video", 0.5))
                a_tok = a_tok * a_w
                v_tok = v_tok * v_w
            else:
                # tensor (B,2): [a,v]
                a_w = alpha[:,0].view(B,1,1)
                v_w = alpha[:,1].view(B,1,1)
                a_tok = a_tok * a_w
                v_tok = v_tok * v_w

        prefix = torch.cat([a_tok, v_tok], dim=1)  # (B, K, D)
        K = prefix.shape[1]

        inputs_embeds = torch.cat([prefix, emb], dim=1)
        # attention mask: prefix all ones
        prefix_mask = torch.ones((B, K), dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        # labels: ignore prefix
        ignore = torch.full((B, K), -100, dtype=labels.dtype, device=labels.device)
        lbl = torch.cat([ignore, labels], dim=1)

        inputs_embeds = inputs_embeds.to(self.lm.dtype)
        out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn, labels=lbl, return_dict=True)
        loss = out.loss

        # sample weighting (scalar per sample)
        if sample_weight is not None:
            # Transformers loss is averaged; we approximate by scaling loss by mean weight
            w = sample_weight.float().mean()
            loss = loss * w

        aux_loss = None
        if self.mu_head is not None and mu_target is not None:
            # use first prefix token as a summary representation (audio first token)
            rep = prefix[:,0,:]  # (B,D)
            mu_hat = self.mu_head(rep)
            aux_loss = torch.nn.functional.mse_loss(mu_hat, mu_target.float())
            loss = loss + 0.1 * aux_loss

        return {"loss": loss, "lm_loss": out.loss, "aux_loss": aux_loss}
