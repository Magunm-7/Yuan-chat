from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import SoftTokenProjector


class MultiModalPrefixLM(nn.Module):
    """
    LLaMA + (audio/video)->prefix soft tokens.
    Freeze base LM by default; train projectors (and optional small head).
    """
    def __init__(
        self,
        base_lm,
        d_model: int,
        audio_c: int,
        video_c: int,
        k_audio: int = 8,
        k_video: int = 8,
        projector_hidden: int = 512,
        train_base: bool = False,
        use_alpha_gate: bool = True,
        aux_mu_dim: int = 0,
    ):
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

        # Freeze base lm (but keep LoRA trainable if present)
        if not train_base:
            for n, p in self.lm.named_parameters():
                # common PEFT naming patterns
                is_lora = ("lora_" in n) or (".lora_A" in n) or (".lora_B" in n)
                p.requires_grad_(bool(is_lora))

        # Ensure projectors / head are trainable
        for p in self.audio_proj.parameters():
            p.requires_grad_(True)
        for p in self.video_proj.parameters():
            p.requires_grad_(True)
        if self.mu_head is not None:
            for p in self.mu_head.parameters():
                p.requires_grad_(True)

    def _apply_alpha_gate(self, a_tok, v_tok, alpha):
        """
        alpha:
          - tensor (B,2): [a,v]
          - dict:
              {"audio": float, "video": float}  (global)
              or {"audio": [..B..], "video": [..B..]} (per-sample)
        """
        if alpha is None:
            return a_tok, v_tok

        B = a_tok.shape[0]
        device = a_tok.device
        dtype = a_tok.dtype

        if isinstance(alpha, dict):
            a = alpha.get("audio", 0.5)
            v = alpha.get("video", 0.5)
            # allow scalar or list/1d-tensor
            if not torch.is_tensor(a):
                a = torch.tensor(a, device=device, dtype=dtype)
            if not torch.is_tensor(v):
                v = torch.tensor(v, device=device, dtype=dtype)
            if a.ndim == 0:
                a = a.view(1).repeat(B)
            if v.ndim == 0:
                v = v.view(1).repeat(B)
            a_w = a.view(B, 1, 1)
            v_w = v.view(B, 1, 1)
        else:
            # tensor (B,2)
            a_w = alpha[:, 0].view(B, 1, 1).to(device=device, dtype=dtype)
            v_w = alpha[:, 1].view(B, 1, 1).to(device=device, dtype=dtype)

        return a_tok * a_w, v_tok * v_w

    def forward(
        self,
        input_ids,
        attention_mask,
        labels,
        audio_feat,
        video_feat,
        alpha=None,
        sample_weight=None,
        mu_target=None,
    ):
        """
        audio_feat: (B, Ca) pooled
        video_feat: (B, Cv) pooled
        alpha: (B,2) or dict
        """
        B = input_ids.shape[0]

        # token embeddings
        emb = self.lm.get_input_embeddings()(input_ids)

        # prefix soft tokens
        a_tok = self.audio_proj(audio_feat)  # (B,Ka,D)
        v_tok = self.video_proj(video_feat)  # (B,Kv,D)

        if self.use_alpha_gate and alpha is not None:
            a_tok, v_tok = self._apply_alpha_gate(a_tok, v_tok, alpha)

        prefix = torch.cat([a_tok, v_tok], dim=1)  # (B,K,D)
        K = prefix.shape[1]

        inputs_embeds = torch.cat([prefix, emb], dim=1)
        prefix_mask = torch.ones((B, K), dtype=attention_mask.dtype, device=attention_mask.device)
        attn = torch.cat([prefix_mask, attention_mask], dim=1)

        ignore = torch.full((B, K), -100, dtype=labels.dtype, device=labels.device)
        lbl = torch.cat([ignore, labels], dim=1)

        # keep dtype boundary explicit
        inputs_embeds = inputs_embeds.to(self.lm.dtype)

        # IMPORTANT: per-token loss for correct per-sample weighting
        out = self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            labels=lbl,
            return_dict=True,
        )

        # out.loss is already averaged; we recompute token loss per sample for proper weighting
        # logits: (B, T, V); labels: (B, T)
        logits = out.logits
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = lbl[:, 1:].contiguous()

        # token-level CE (no reduction)
        token_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, -1)  # (B, T-1)

        token_mask = (shift_labels != -100).float()  # (B, T-1)
        denom = token_mask.sum(dim=1).clamp_min(1.0)  # (B,)
        per_sample_loss = (token_loss * token_mask).sum(dim=1) / denom  # (B,)

        if sample_weight is not None:
            w = sample_weight.float().view(B)  # (B,)
            per_sample_loss = per_sample_loss * w

        loss = per_sample_loss.mean()

        aux_loss = None
        if self.mu_head is not None and mu_target is not None:
            rep = prefix[:, 0, :]  # (B,D)
            mu_hat = self.mu_head(rep)
            aux_loss = F.mse_loss(mu_hat, mu_target.float())
            loss = loss + 0.1 * aux_loss

        return {"loss": loss, "lm_loss": out.loss, "aux_loss": aux_loss}
