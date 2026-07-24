from __future__ import annotations
import json
import numpy as np
import torch
from torch.utils.data import Dataset


def _apply_chat_template(tokenizer, messages, add_generation_prompt: bool):
    """
    Returns rendered string using tokenizer chat template if available,
    otherwise falls back to a simple ROLE: content format.

    add_generation_prompt=True means we end at the assistant header (no assistant content),
    so its token length can be used to mask labels.
    """
    if hasattr(tokenizer, "apply_chat_template"):
        # Qwen3: force non-thinking (no <think> block) for short MI replies.
        # Qwen2.5's template has no such flag; it simply ignores the extra kwarg.
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=add_generation_prompt,
            )

    # fallback: simple text format
    parts = []
    for m in messages:
        parts.append(f"{m['role'].upper()}: {m['content']}".strip())

    if add_generation_prompt:
        # add assistant header only (no content)
        parts.append("ASSISTANT:")
        return "\n".join(parts) + " "
    else:
        return "\n".join(parts) + "\n"


def _encode_text(tokenizer, text: str, max_len: int):
    enc = tokenizer(
        text,
        truncation=True,
        max_length=max_len,
        return_tensors="pt",
        padding=False,
    )
    return enc["input_ids"][0], enc["attention_mask"][0]


def _build_prompt_and_full(tokenizer, messages, max_len: int):
    """
    We want to train only assistant tokens.
    Strategy:
      - prompt_text: system+user (+ assistant header), add_generation_prompt=True
      - full_text:  system+user+assistant, add_generation_prompt=False
      - labels: full_ids, but labels[:len(prompt_ids)] = -100
    """
    # target = the LAST assistant turn; everything before it is prompt (history assistants stay masked)
    last_idx = -1
    for j, m in enumerate(messages):
        if m.get("role") == "assistant":
            last_idx = j

    if last_idx < 0:
        # no assistant message -> train nothing (all -100)
        full_text = _apply_chat_template(tokenizer, messages, add_generation_prompt=False)
        full_ids, full_attn = _encode_text(tokenizer, full_text, max_len)
        labels = full_ids.clone()
        labels[:] = -100
        return full_ids, full_attn, labels

    prefix_msgs = messages[:last_idx]
    full_msgs = messages[:last_idx + 1]
    prompt_text = _apply_chat_template(tokenizer, prefix_msgs, add_generation_prompt=True)
    full_text = _apply_chat_template(tokenizer, full_msgs, add_generation_prompt=False)

    # true (untruncated) lengths -> exact target token count = full - prompt
    full_full = tokenizer(full_text, return_tensors="pt", truncation=False)["input_ids"][0]
    prompt_full = tokenizer(prompt_text, return_tensors="pt", truncation=False)["input_ids"][0]
    n_target = max(1, int(full_full.shape[0]) - int(prompt_full.shape[0]))

    full_ids = full_full[-max_len:]           # LEFT-truncate: drop oldest history, keep target at the end
    full_attn = torch.ones_like(full_ids)
    labels = full_ids.clone()
    n_mask = int(full_ids.shape[0]) - n_target
    if n_mask > 0:
        labels[:n_mask] = -100                # mask everything except the last n_target (the reply)
    return full_ids, full_attn, labels


def _safe_load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


class MMCacheDataset(Dataset):
    """
    Each item points to a .npz containing audio/video pooled features and metadata,
    plus a json record that holds messages and weight.
    """
    def __init__(self, index_jsonl: str, tokenizer, max_len: int = 1024):
        self.items = _safe_load_jsonl(index_jsonl)
        self.tok = tokenizer
        # truncate from the LEFT so long dialogues drop the OLDEST history, never the target reply
        self.tok.truncation_side = "left"
        self.max_len = max_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        npz = np.load(it["npz_path"])
        alpha = npz["alpha"].astype(np.float32)       # (2,)
        mu = npz["mu"].astype(np.float32)             # (M,)

        messages = it["messages"]
        input_ids, attn, labels = _build_prompt_and_full(self.tok, messages, self.max_len)

        out = {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "alpha": torch.from_numpy(alpha),
            "mu": torch.from_numpy(mu),
            "sample_weight": torch.tensor(float(it.get("sample_weight", 1.0)), dtype=torch.float32),
        }
        if "audio_seq" in npz:   # cross-attention path: sequences + text embedding
            out["audio_seq"] = torch.from_numpy(npz["audio_seq"].astype(np.float32))
            out["video_seq"] = torch.from_numpy(npz["video_seq"].astype(np.float32))
            out["text_emb"] = torch.from_numpy(npz["text_emb"].astype(np.float32))
        else:                    # pooled path
            out["audio_feat"] = torch.from_numpy(npz["audio_feat"].astype(np.float32))
            out["video_feat"] = torch.from_numpy(npz["video_feat"].astype(np.float32))
        return out


def collate_mm(batch, pad_token_id: int | None = None):
    # pad to max len in batch
    max_len = max(x["input_ids"].shape[0] for x in batch)

    if pad_token_id is None:
        # best-effort: derive from batch dtype (we'll default to 0 only if truly unknown)
        pad_token_id = 0

    def pad1d(x, pad_val):
        if x.shape[0] == max_len:
            return x
        pad = torch.full((max_len - x.shape[0],), pad_val, dtype=x.dtype)
        return torch.cat([x, pad], dim=0)

    input_ids = torch.stack([pad1d(x["input_ids"], pad_val=pad_token_id) for x in batch], dim=0)
    attention_mask = torch.stack([pad1d(x["attention_mask"], pad_val=0) for x in batch], dim=0)
    labels = torch.stack([pad1d(x["labels"], pad_val=-100) for x in batch], dim=0)

    alpha = torch.stack([x["alpha"] for x in batch], dim=0)
    mu = torch.stack([x["mu"] for x in batch], dim=0)
    w = torch.stack([x["sample_weight"] for x in batch], dim=0)

    out = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        alpha=alpha,
        mu=mu,
        sample_weight=w,
    )
    if "audio_seq" in batch[0]:
        out["audio_seq"] = torch.stack([x["audio_seq"] for x in batch], dim=0)
        out["video_seq"] = torch.stack([x["video_seq"] for x in batch], dim=0)
        out["text_emb"] = torch.stack([x["text_emb"] for x in batch], dim=0)
    else:
        out["audio_feat"] = torch.stack([x["audio_feat"] for x in batch], dim=0)
        out["video_feat"] = torch.stack([x["video_feat"] for x in batch], dim=0)
    return out
