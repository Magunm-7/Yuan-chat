from __future__ import annotations
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .projector import SoftTokenProjector
from .fusion import CrossAttnFusion


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
        use_state_prefix: bool = False,
        state_dim: int = 0,
        k_state: int = 4,
        use_crossattn: bool = False,
        text_dim: int = 384,
        k_fuse: int = 8,
        d_fuse: int = 512,
    ):
        super().__init__()
        self.lm = base_lm
        self.d_model = d_model
        self.use_alpha_gate = use_alpha_gate
        self.use_state_prefix = use_state_prefix
        self.use_crossattn = use_crossattn

        if use_crossattn:
            # route 2: text-queried cross-attention over audio/video SEQUENCES (audio_c/video_c = seq feat dim)
            self.fusion = CrossAttnFusion(d_model, text_dim=text_dim, audio_dim=audio_c,
                                          video_dim=video_c, k_tokens=k_fuse, d_fuse=d_fuse)
            self.audio_proj = None
            self.video_proj = None
        else:
            self.fusion = None
            self.audio_proj = SoftTokenProjector(audio_c, d_model, k_tokens=k_audio, hidden=projector_hidden)
            self.video_proj = SoftTokenProjector(video_c, d_model, k_tokens=k_video, hidden=projector_hidden)
        # route A->B: inject the evaluator's state mu as a soft-token prefix (state-conditioned generation)
        if use_state_prefix and state_dim > 0:
            self.state_proj = SoftTokenProjector(state_dim, d_model, k_tokens=k_state, hidden=projector_hidden)
        else:
            self.state_proj = None

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

        # Ensure fusion / projectors / heads are trainable
        for mod in (self.fusion, self.audio_proj, self.video_proj, self.state_proj, self.mu_head):
            if mod is not None:
                for p in mod.parameters():
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
        audio_feat=None,
        video_feat=None,
        alpha=None,
        sample_weight=None,
        mu_target=None,
        state=None,
        text_emb=None,
        audio_seq=None,
        video_seq=None,
    ):
        """
        pooled path : audio_feat/video_feat (B, C)
        crossattn   : text_emb (B, Ct), audio_seq (B, Ta, C), video_seq (B, Tv, C)
        alpha: (B,2) modality gate; state: (B, state_dim) evaluator mu (state prefix)
        """
        B = input_ids.shape[0]

        # token embeddings
        emb = self.lm.get_input_embeddings()(input_ids)

        # multimodal prefix soft tokens
        if self.use_crossattn:
            mm_tok = self.fusion(text_emb, audio_seq, video_seq,
                                 alpha if self.use_alpha_gate else None)   # (B,Kf,D)
            toks = [mm_tok]
        else:
            a_tok = self.audio_proj(audio_feat)  # (B,Ka,D)
            v_tok = self.video_proj(video_feat)  # (B,Kv,D)
            if self.use_alpha_gate and alpha is not None:
                a_tok, v_tok = self._apply_alpha_gate(a_tok, v_tok, alpha)
            toks = [a_tok, v_tok]
        if self.state_proj is not None and state is not None:
            toks = [self.state_proj(state)] + toks   # evaluator state tokens go first
        prefix = torch.cat(toks, dim=1)  # (B,K,D)
        K = prefix.shape[1]

        # text_only(alpha=0)下 prefix 恒为全零:不携带任何信息,却会让 RMSNorm 的反向撞上
        # variance=0 的数值奇点。Qwen3 因多一层 QK-Norm 尤其敏感——实测全零 prefix 导致
        # 232/320 个梯度 NaN,而同样 16 个 token 换成随机小值、或直接不拼,梯度都干净。
        # 全零时跳过拼接:语义完全等价,还省掉 K 个位置的注意力计算。
        if bool((prefix != 0).any()):
            inputs_embeds = torch.cat([prefix, emb], dim=1)
            prefix_mask = torch.ones((B, K), dtype=attention_mask.dtype, device=attention_mask.device)
            attn = torch.cat([prefix_mask, attention_mask], dim=1)
            ignore = torch.full((B, K), -100, dtype=labels.dtype, device=labels.device)
            lbl = torch.cat([ignore, labels], dim=1)
        else:
            inputs_embeds, attn, lbl = emb, attention_mask, labels

        # keep dtype boundary explicit
        inputs_embeds = inputs_embeds.to(self.lm.dtype)

        # Only the trailing reply is supervised, but a full forward materialises
        # (T x vocab) logits -- at 14B/151k-vocab that dominates activation memory.
        # With B=1 there is no right padding, so the target is the tail and we can ask
        # for just the last n_target+1 positions instead of all T.
        n_keep = None
        if B == 1 and os.environ.get("MPSE_NO_FAST_LOGITS") != "1":
            n_tgt = int((lbl != -100).sum().item())
            if n_tgt > 0:
                n_keep = min(n_tgt + 1, int(lbl.shape[1]))

        out = None
        if n_keep is not None:
            for kw in ("logits_to_keep", "num_logits_to_keep"):
                try:
                    out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn,
                                  return_dict=True, **{kw: n_keep})
                    break
                except TypeError:
                    out = None
            # the kwarg may be silently swallowed by **kwargs -> verify it actually took effect
            if out is not None and int(out.logits.shape[1]) != n_keep:
                out, n_keep = None, None

        if out is None:  # older transformers, or B>1 (right padding breaks the tail assumption)
            n_keep = None
            out = self.lm(inputs_embeds=inputs_embeds, attention_mask=attn,
                          labels=lbl, return_dict=True)

        # per-token loss (not out.loss) so each sample can be weighted by sigma
        logits = out.logits
        if n_keep is not None:
            shift_logits = logits[:, :-1, :]                 # predicts the target tokens
            shift_labels = lbl[:, -(n_keep - 1):]
        else:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = lbl[:, 1:].contiguous()

        # token-level CE (no reduction)
        token_loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, -1)  # (B, T-1) or (B, n_target)

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

        return {"loss": loss, "lm_loss": getattr(out, "loss", None), "aux_loss": aux_loss}
