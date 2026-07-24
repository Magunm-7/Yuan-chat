from __future__ import annotations
import os, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM


def _pick_torch_dtype(device: str) -> torch.dtype:
    """
    Prefer bf16 on modern GPUs (more stable than fp16),
    fallback to fp16 on cuda, else fp32 on cpu.
    """
    if not device.startswith("cuda"):
        return torch.float32
    if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train_mm_sft(
    index_jsonl: str,
    base_model_dir: str,
    out_dir: str,
    batch_size: int = 1,
    lr: float = 2e-4,
    epochs: int = 1,
    k_audio: int = 8,
    k_video: int = 8,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    max_len: int = 1024,
    aux_mu: bool = True,
    text_only: bool = False,   # control baseline: zero the alpha gate -> prefix carries no modality signal
    use_state_prefix: bool = False,   # route A->B: inject evaluator mu as prefix soft tokens
    k_state: int = 4,
    k_fuse: int = 8,                  # cross-attention fusion output tokens (auto-enabled when npz has sequences)
    gradient_checkpointing: bool = False,   # trade compute for memory (needed for long seqs / dialogue history)
    # LoRA args
    use_lora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj"),
):
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM

    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    torch_dtype = _pick_torch_dtype(device)

    lm = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        torch_dtype=torch_dtype if device.startswith("cuda") else torch.float32,
    )
    lm.to(device)

    if gradient_checkpointing:
        lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- LoRA injection (fine-tune base via adapters) ----
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model
        except Exception as e:
            raise RuntimeError(
                "use_lora=True but 'peft' is not available. Please install peft."
            ) from e

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(lora_target_modules),
        )
        lm = get_peft_model(lm, lora_cfg)

    if gradient_checkpointing:
        lm.enable_input_require_grads()

    # IMPORTANT:
    # - projector is trained anyway
    # - LoRA params should be trainable
    # - base weights stay frozen (your model_wrap has been fixed to not freeze LoRA)
    lm.train()

    # ---- infer dims from cache first line (safe) ----
    with open(index_jsonl, "r", encoding="utf-8") as f:
        first_line = f.readline().strip()
    if not first_line:
        raise RuntimeError(f"index_jsonl is empty: {index_jsonl}")

    first = json.loads(first_line)
    npz = __import__("numpy").load(first["npz_path"])
    use_crossattn = "audio_seq" in npz            # data decides: sequences -> cross-attention fusion
    if use_crossattn:
        audio_c = int(npz["audio_seq"].shape[-1])
        video_c = int(npz["video_seq"].shape[-1])
        text_dim = int(npz["text_emb"].shape[0])
    else:
        audio_c = int(npz["audio_feat"].shape[0])
        video_c = int(npz["video_feat"].shape[0])
        text_dim = 384
    d_model = int(lm.config.hidden_size)
    mu_dim = int(npz["mu"].shape[0]) if aux_mu else 0
    state_dim = int(npz["mu"].shape[0]) if use_state_prefix else 0

    model = MultiModalPrefixLM(
        lm,
        d_model=d_model,
        audio_c=audio_c,
        video_c=video_c,
        k_audio=k_audio,
        k_video=k_video,
        projector_hidden=512,
        train_base=False,          # keep "base frozen", LoRA stays trainable after your model_wrap fix
        use_alpha_gate=True,
        aux_mu_dim=mu_dim,
        use_state_prefix=use_state_prefix,
        state_dim=state_dim,
        k_state=k_state,
        use_crossattn=use_crossattn,
        text_dim=text_dim,
        k_fuse=k_fuse,
    )
    model.to(device)
    model.train()

    ds = MMCacheDataset(index_jsonl, tokenizer=tok, max_len=max_len)
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_mm(b, pad_token_id=tok.pad_token_id),
        pin_memory=device.startswith("cuda"),
    )

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    use_amp = device.startswith("cuda")
    # GradScaler only needed for fp16; bf16 usually无需 scaler
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and torch_dtype == torch.float16))

    for ep in range(1, epochs + 1):
        pbar = tqdm(dl, desc=f"MM-SFT ep {ep}/{epochs}")
        for batch in pbar:
            # move tensors safely (avoid KeyError)
            for k in ["input_ids","attention_mask","labels","audio_feat","video_feat","text_emb","audio_seq","video_seq","alpha","mu","sample_weight"]:
                if k in batch and torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device, non_blocking=True)
            if text_only:
                batch["alpha"] = torch.zeros_like(batch["alpha"])  # ablate multimodal signal

            with torch.autocast(
                device_type="cuda" if use_amp else "cpu",
                dtype=torch_dtype if torch_dtype in (torch.float16, torch.bfloat16) else torch.float16,
                enabled=use_amp,
            ):
                out = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                    audio_feat=batch.get("audio_feat"),
                    video_feat=batch.get("video_feat"),
                    text_emb=batch.get("text_emb"),
                    audio_seq=batch.get("audio_seq"),
                    video_seq=batch.get("video_seq"),
                    alpha=batch["alpha"],
                    sample_weight=batch.get("sample_weight", None),
                    mu_target=batch["mu"] if (mu_dim > 0 and "mu" in batch) else None,
                    state=batch["mu"] if (use_state_prefix and "mu" in batch) else None,
                )
                loss = out["loss"]

            opt.zero_grad(set_to_none=True)

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()

            pbar.set_postfix(loss=float(loss.detach().cpu()))

    # ---- Save projector/prefix ckpt ----
    ckpt = os.path.join(out_dir, "mm_prefix.pt")
    torch.save({
        "audio_proj": model.audio_proj.state_dict() if model.audio_proj is not None else None,
        "video_proj": model.video_proj.state_dict() if model.video_proj is not None else None,
        "fusion": model.fusion.state_dict() if model.fusion is not None else None,
        "use_crossattn": bool(use_crossattn), "text_dim": text_dim, "k_fuse": k_fuse,
        "mu_head": model.mu_head.state_dict() if model.mu_head is not None else None,
        "k_audio": k_audio, "k_video": k_video,
        "audio_c": audio_c, "video_c": video_c,
        "d_model": d_model,
        "mu_dim": mu_dim,
        "use_lora": bool(use_lora),
        "lora_target_modules": list(lora_target_modules),
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        # extra
        "base_model_dir": base_model_dir,
        "max_len": max_len,
        "torch_dtype": str(torch_dtype),
        "text_only": bool(text_only),
        "use_state_prefix": bool(use_state_prefix),
        "state_dim": state_dim,
        "k_state": k_state,
        "state_proj": model.state_proj.state_dict() if model.state_proj is not None else None,
    }, ckpt)

    # ---- Save LoRA adapter (if enabled) ----
    if use_lora:
        lora_dir = os.path.join(out_dir, "lora_adapter")
        os.makedirs(lora_dir, exist_ok=True)
        # model.lm is the peft-wrapped lm inside MultiModalPrefixLM
        model.lm.save_pretrained(lora_dir)
        tok.save_pretrained(lora_dir)

    return ckpt


def load_mm_prefix(lm, ckpt_path: str, device: str):
    """
    Backward-compatible loader:
    - loads projector weights from mm_prefix.pt
    - if a sibling directory 'lora_adapter/' exists, it will also load LoRA onto lm
    """
    sd = torch.load(ckpt_path, map_location="cpu")

    # auto-load LoRA if present
    lora_dir = os.path.join(os.path.dirname(ckpt_path), "lora_adapter")
    if os.path.isdir(lora_dir):
        try:
            from peft import PeftModel
            lm = PeftModel.from_pretrained(lm, lora_dir)
        except Exception as e:
            raise RuntimeError(f"Found lora_adapter at {lora_dir} but failed to load it.") from e

    model = MultiModalPrefixLM(
        lm,
        d_model=sd["d_model"],
        audio_c=sd["audio_c"],
        video_c=sd["video_c"],
        k_audio=sd["k_audio"],
        k_video=sd["k_video"],
        projector_hidden=512,
        train_base=False,
        use_alpha_gate=True,
        aux_mu_dim=sd.get("mu_dim", 0),
        use_state_prefix=sd.get("use_state_prefix", False),
        state_dim=sd.get("state_dim", 0),
        k_state=sd.get("k_state", 4),
        use_crossattn=sd.get("use_crossattn", False),
        text_dim=sd.get("text_dim", 384),
        k_fuse=sd.get("k_fuse", 8),
    )
    if model.audio_proj is not None and sd.get("audio_proj") is not None:
        model.audio_proj.load_state_dict(sd["audio_proj"])
    if model.video_proj is not None and sd.get("video_proj") is not None:
        model.video_proj.load_state_dict(sd["video_proj"])
    if model.fusion is not None and sd.get("fusion") is not None:
        model.fusion.load_state_dict(sd["fusion"])
    if model.state_proj is not None and sd.get("state_proj") is not None:
        model.state_proj.load_state_dict(sd["state_proj"])
    if model.mu_head is not None and sd.get("mu_head") is not None:
        model.mu_head.load_state_dict(sd["mu_head"])
    model.to(device)
    model.eval()
    return model
