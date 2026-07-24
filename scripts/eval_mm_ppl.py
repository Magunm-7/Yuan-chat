"""
B-stage eval: held-out perplexity on the therapist reply (assistant tokens only).

One config per run (compose by calling 3x):
  base        : raw base model, no adapter, no prefix         (floor)
  text-only   : --lora DIR --mm_prefix DIR/mm_prefix.pt --zero_alpha
  multimodal  : --lora DIR --mm_prefix DIR/mm_prefix.pt

text-only vs multimodal differ ONLY in whether the audio/video signal flows
(zero_alpha silences the prefix), so the ppl gap isolates the multimodal effect.

  python scripts/eval_mm_ppl.py --base <BASE> --index data/annomi/mm_sft/holdout.jsonl \
      --lora outputs/mm_sft/qwen7b_mm --mm_prefix outputs/mm_sft/qwen7b_mm/mm_prefix.pt
"""
from __future__ import annotations
import os, argparse, math, json
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.train_mm_sft import load_mm_prefix, _pick_torch_dtype


@torch.no_grad()
def _target_logits(lm, labels, **fwd):
    """
    Returns (shift_logits, shift_labels) aligned for next-token CE.

    Full (T x vocab) logits dominate activation memory at 14B scale, while only the
    trailing reply is scored. With batch=1 there is no right padding, so the target
    is the tail and we can ask the LM for just the last n_target+1 positions.
    """
    n_tgt = int((labels != -100).sum().item())
    n_keep = min(n_tgt + 1, int(labels.shape[1])) if n_tgt > 0 else 0
    if int(labels.shape[0]) == 1 and n_keep > 1 and os.environ.get("MPSE_NO_FAST_LOGITS") != "1":
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = lm(**fwd, return_dict=True, **{kw: n_keep})
            except TypeError:
                continue
            if int(out.logits.shape[1]) == n_keep:      # kwarg honoured
                return out.logits[:, :-1, :], labels[:, -(n_keep - 1):]
            break                                        # swallowed by **kwargs -> fall back
    out = lm(**fwd, return_dict=True)
    return out.logits[:, :-1, :], labels[:, 1:]


@torch.no_grad()
def _assistant_nll(shift_logits, shift_labels):
    """Summed CE over non-masked (assistant) tokens; returns (nll_sum, n_tokens)."""
    nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)).float(),
                          shift_labels.reshape(-1),
                          reduction="sum", ignore_index=-100)
    n = int((shift_labels != -100).sum())
    return float(nll), n


@torch.no_grad()
def eval_ppl(index, base, lora=None, mm_prefix=None, zero_alpha=False,
             max_len=512, dump=None, device="cuda" if torch.cuda.is_available() else "cpu"):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = _pick_torch_dtype(device)
    lm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=dtype).to(device).eval()

    mm = None
    if mm_prefix:                      # builds MultiModalPrefixLM + auto-loads sibling lora_adapter
        mm = load_mm_prefix(lm, mm_prefix, device)
        mm.eval()
    elif lora:                         # text-only LoRA, no prefix
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, lora).to(device).eval()

    ds = MMCacheDataset(index, tokenizer=tok, max_len=max_len)
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    collate_fn=lambda b: collate_mm(b, pad_token_id=tok.pad_token_id))

    tot_nll, tot_tok = 0.0, 0
    per_turn = []
    for b in dl:
        for k in ["input_ids", "attention_mask", "labels", "audio_feat", "video_feat",
                  "text_emb", "audio_seq", "video_seq", "alpha", "mu"]:
            if k in b and torch.is_tensor(b[k]):
                b[k] = b[k].to(device)
        # autocast so fp32 projector weights + bf16 model embeddings mix cleanly (as in training)
        with torch.autocast(device_type="cuda", dtype=dtype, enabled=device.startswith("cuda")):
            if mm is not None:
                alpha = torch.zeros_like(b["alpha"]) if zero_alpha else b["alpha"]
                emb = mm.lm.get_input_embeddings()(b["input_ids"])
                if getattr(mm, "use_crossattn", False):
                    mm_tok = mm.fusion(b["text_emb"], b["audio_seq"], b["video_seq"],
                                       alpha if mm.use_alpha_gate else None)
                    toks = [mm_tok]
                else:
                    a_tok = mm.audio_proj(b["audio_feat"])
                    v_tok = mm.video_proj(b["video_feat"])
                    if mm.use_alpha_gate:
                        a_tok, v_tok = mm._apply_alpha_gate(a_tok, v_tok, alpha)
                    toks = [a_tok, v_tok]
                if getattr(mm, "state_proj", None) is not None:
                    toks = [mm.state_proj(b["mu"])] + toks   # inject evaluator state
                prefix = torch.cat(toks, dim=1)
                K = prefix.shape[1]
                # 与 model_wrap 保持一致:全零 prefix(zero_alpha/text_only)不拼接,
                # 否则训练与评估的输入长度不同,ppl 不可比。
                if bool((prefix != 0).any()):
                    inp = torch.cat([prefix, emb], dim=1)
                    attn = torch.cat([torch.ones((1, K), device=device, dtype=b["attention_mask"].dtype),
                                      b["attention_mask"]], dim=1)
                    lbl = torch.cat([torch.full((1, K), -100, device=device, dtype=b["labels"].dtype),
                                     b["labels"]], dim=1)
                else:
                    inp, attn, lbl = emb, b["attention_mask"], b["labels"]
                shift_logits, shift_labels = _target_logits(
                    lm, lbl, inputs_embeds=inp, attention_mask=attn)
            else:
                shift_logits, shift_labels = _target_logits(
                    lm, b["labels"], input_ids=b["input_ids"], attention_mask=b["attention_mask"])
        nll, n = _assistant_nll(shift_logits, shift_labels)
        tot_nll += nll; tot_tok += n
        per_turn.append((nll, n))

    if dump:
        with open(dump, "w", encoding="utf-8") as f:
            for nll, n in per_turn:
                f.write(json.dumps({"nll": nll, "ntok": n}) + "\n")

    ppl = math.exp(tot_nll / max(1, tot_tok))
    return ppl, tot_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/annomi/mm_sft/holdout.jsonl")
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--mm_prefix", default=None)
    ap.add_argument("--zero_alpha", action="store_true")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--dump", default=None, help="write per-turn {nll,ntok} jsonl (for paired bootstrap)")
    args = ap.parse_args()

    tag = ("multimodal" if (args.mm_prefix and not args.zero_alpha)
           else "text-only" if args.mm_prefix else "text-LoRA" if args.lora else "base")
    ppl, ntok = eval_ppl(args.index, args.base, args.lora, args.mm_prefix,
                         args.zero_alpha, args.max_len, dump=args.dump)
    print(f"[{tag:11s}] holdout ppl = {ppl:.3f}  (over {ntok} assistant tokens)")


if __name__ == "__main__":
    main()
