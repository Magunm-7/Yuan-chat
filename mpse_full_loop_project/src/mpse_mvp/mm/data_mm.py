
from __future__ import annotations
import os, json
import numpy as np
import torch
from torch.utils.data import Dataset

def _chatml_from_messages(messages, tokenizer):
    # Very small helper: concatenate roles into a single text.
    # Llama-3.x instruct works reasonably with simple chat format if tokenizer has chat template; we use it if available.
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    # fallback
    parts=[]
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}".strip())
    return "\n".join(parts) + "\n"

class MMCacheDataset(Dataset):
    """
    Each item points to a .npz containing audio/video pooled features and metadata,
    plus a json record that holds messages and weight.
    """
    def __init__(self, index_jsonl: str, tokenizer, max_len: int = 1024):
        self.items = [json.loads(l) for l in open(index_jsonl, encoding="utf-8")]
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        npz = np.load(it["npz_path"])
        audio = npz["audio_feat"].astype(np.float32)  # (Ca,)
        video = npz["video_feat"].astype(np.float32)  # (Cv,)
        alpha = npz["alpha"].astype(np.float32)       # (2,)
        mu = npz["mu"].astype(np.float32)             # (M,)

        messages = it["messages"]
        text = _chatml_from_messages(messages, self.tok)
        enc = self.tok(text, truncation=True, max_length=self.max_len, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        attn = enc["attention_mask"][0]

        # causal LM labels: predict all tokens except padding; keep as-is (tokenizer already includes all)
        labels = input_ids.clone()

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "audio_feat": torch.from_numpy(audio),
            "video_feat": torch.from_numpy(video),
            "alpha": torch.from_numpy(alpha),
            "mu": torch.from_numpy(mu),
            "sample_weight": torch.tensor(float(it.get("sample_weight", 1.0)), dtype=torch.float32),
        }

def collate_mm(batch):
    # pad to max len in batch
    max_len = max(x["input_ids"].shape[0] for x in batch)
    def pad1d(x, pad_val):
        if x.shape[0] == max_len:
            return x
        pad = torch.full((max_len - x.shape[0],), pad_val, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)
    input_ids = torch.stack([pad1d(x["input_ids"], pad_val=0) for x in batch], dim=0)
    attention_mask = torch.stack([pad1d(x["attention_mask"], pad_val=0) for x in batch], dim=0)
    labels = torch.stack([pad1d(x["labels"], pad_val=-100) for x in batch], dim=0)

    audio_feat = torch.stack([x["audio_feat"] for x in batch], dim=0)
    video_feat = torch.stack([x["video_feat"] for x in batch], dim=0)
    alpha = torch.stack([x["alpha"] for x in batch], dim=0)
    mu = torch.stack([x["mu"] for x in batch], dim=0)
    w = torch.stack([x["sample_weight"] for x in batch], dim=0)

    return dict(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                audio_feat=audio_feat, video_feat=video_feat, alpha=alpha, mu=mu, sample_weight=w)
