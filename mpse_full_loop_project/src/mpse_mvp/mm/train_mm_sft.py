
from __future__ import annotations
import os, json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM

def train_mm_sft(index_jsonl: str, base_model_dir: str, out_dir: str,
                 batch_size: int = 1, lr: float = 2e-4, epochs: int = 1,
                 k_audio: int = 8, k_video: int = 8,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 max_len: int = 1024,
                 aux_mu: bool = True):
    os.makedirs(out_dir, exist_ok=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base_model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    lm.to(device)
    lm.eval()

    # infer dims from cache first line
    first = json.loads(open(index_jsonl, encoding="utf-8").readline())
    npz = __import__("numpy").load(first["npz_path"])
    audio_c = int(npz["audio_feat"].shape[0])
    video_c = int(npz["video_feat"].shape[0])
    d_model = int(lm.config.hidden_size)
    mu_dim = int(npz["mu"].shape[0]) if aux_mu else 0

    model = MultiModalPrefixLM(lm, d_model=d_model, audio_c=audio_c, video_c=video_c,
                              k_audio=k_audio, k_video=k_video, projector_hidden=512,
                              train_base=False, use_alpha_gate=True, aux_mu_dim=mu_dim)
    model.to(device)
    model.train()

    ds = MMCacheDataset(index_jsonl, tokenizer=tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=collate_mm)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    for ep in range(1, epochs+1):
        pbar = tqdm(dl, desc=f"MM-SFT ep {ep}/{epochs}")
        for batch in pbar:
            for k in ["input_ids","attention_mask","labels","audio_feat","video_feat","alpha","mu","sample_weight"]:
                batch[k] = batch[k].to(device)
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
                audio_feat=batch["audio_feat"],
                video_feat=batch["video_feat"],
                alpha=batch["alpha"],
                sample_weight=batch["sample_weight"],
                mu_target=batch["mu"] if mu_dim>0 else None
            )
            loss = out["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            pbar.set_postfix(loss=float(loss.detach().cpu()))

    ckpt = os.path.join(out_dir, "mm_prefix.pt")
    torch.save({
        "audio_proj": model.audio_proj.state_dict(),
        "video_proj": model.video_proj.state_dict(),
        "mu_head": model.mu_head.state_dict() if model.mu_head is not None else None,
        "k_audio": k_audio, "k_video": k_video,
        "audio_c": audio_c, "video_c": video_c,
        "d_model": d_model,
        "mu_dim": mu_dim,
    }, ckpt)
    return ckpt

def load_mm_prefix(lm, ckpt_path: str, device: str):
    import numpy as np
    sd = torch.load(ckpt_path, map_location="cpu")
    model = MultiModalPrefixLM(lm, d_model=sd["d_model"], audio_c=sd["audio_c"], video_c=sd["video_c"],
                              k_audio=sd["k_audio"], k_video=sd["k_video"],
                              projector_hidden=512, train_base=False, use_alpha_gate=True,
                              aux_mu_dim=sd.get("mu_dim",0))
    model.audio_proj.load_state_dict(sd["audio_proj"])
    model.video_proj.load_state_dict(sd["video_proj"])
    if model.mu_head is not None and sd.get("mu_head") is not None:
        model.mu_head.load_state_dict(sd["mu_head"])
    model.to(device)
    model.eval()
    return model
