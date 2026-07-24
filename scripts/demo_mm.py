import json, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mpse_mvp.mm.train_mm_sft import load_mm_prefix

def chat_text(tok, messages):
    # 用 chat template（如果有）
    if hasattr(tok, "apply_chat_template"):
        # enable_thinking=False 让生成侧 prompt 与训练完全一致
        # (Qwen3 会在这里放一个空 <think> 块; Qwen2.5 忽略该参数)
        try:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                           enable_thinking=False)
        except TypeError:
            return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback
    s=[]
    for m in messages:
        s.append(f"{m['role'].upper()}: {m['content']}".strip())
    return "\n".join(s) + "\nASSISTANT:"

@torch.no_grad()
def build_inputs_embeds(mm, input_ids, attention_mask, alpha, state=None,
                        audio_feat=None, video_feat=None,
                        text_emb=None, audio_seq=None, video_seq=None):
    B = input_ids.size(0)
    emb = mm.lm.get_input_embeddings()(input_ids)

    if getattr(mm, "use_crossattn", False):
        mm_tok = mm.fusion(text_emb, audio_seq, video_seq, alpha if mm.use_alpha_gate else None)
        toks = [mm_tok]
    else:
        a_tok = mm.audio_proj(audio_feat)
        v_tok = mm.video_proj(video_feat)
        if mm.use_alpha_gate and alpha is not None:
            a_tok = a_tok * alpha[:, 0].view(B, 1, 1)
            v_tok = v_tok * alpha[:, 1].view(B, 1, 1)
        toks = [a_tok, v_tok]
    if getattr(mm, "state_proj", None) is not None and state is not None:
        toks = [mm.state_proj(state)] + toks   # inject the (possibly overridden) evaluator state
    prefix = torch.cat(toks, dim=1).to(emb.dtype)  # match emb dtype for the cat
    K = prefix.size(1)

    inputs_embeds = torch.cat([prefix, emb], dim=1)
    inputs_embeds = inputs_embeds.to(mm.lm.dtype)

    prefix_mask = torch.ones((B, K), dtype=attention_mask.dtype, device=attention_mask.device)
    attn = torch.cat([prefix_mask, attention_mask], dim=1)

    # 给 generate 一个“同长度”的 dummy input_ids（占位即可）
    dummy = torch.full((B, K), fill_value=0, dtype=input_ids.dtype, device=input_ids.device)
    full_ids = torch.cat([dummy, input_ids], dim=1)

    return full_ids, inputs_embeds, attn, K

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model_dir", required=True)
    ap.add_argument("--mm_prefix", required=True)
    ap.add_argument("--index_jsonl", required=True)
    # 可选：显式指定 LoRA adapter 目录（如果你不想依赖 auto-detect / load_mm_prefix 的行为）
    ap.add_argument("--lora_adapter", default=None)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--greedy", action="store_true", help="greedy decode (collapses to backchannels; sampling is default)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min_gold_words", type=int, default=0, help="only show turns whose gold reply has >= this many words")
    ap.add_argument("--zero_alpha", action="store_true", help="silence the prefix (text-only demo)")
    ap.add_argument("--sweep_dim", default=None, choices=["chg", "aro", "val"],
                    help="state-controllability: hold input fixed, sweep this state dim")
    ap.add_argument("--sweep_vals", default="0.1,0.9", help="comma values to set --sweep_dim to")
    ap.add_argument("--diag_av", action="store_true", help="diagnostic: real vs zeroed audio/video prefix (is the prefix used at all?)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    ).to(args.device)

    # 如果你传了 lora_adapter，就在这里手动加载（更确定）
    # 不传也没关系：后面的 load_mm_prefix 可能会自动加载同目录下的 lora_adapter
    if args.lora_adapter is not None:
        from peft import PeftModel
        lm = PeftModel.from_pretrained(lm, args.lora_adapter)

    lm.eval()

    # load_mm_prefix：加载 projector；如果你的 load_mm_prefix 实现会 auto-load lora_adapter，也会在这里生效
    mm = load_mm_prefix(lm, args.mm_prefix, device=args.device)
    mm.eval()

    # 关键：生成时用 mm.lm（确保 LoRA 生效），而不是用最开始的 lm
    gen_lm = mm.lm
    gen_lm.eval()

    items = [json.loads(l) for l in open(args.index_jsonl, encoding="utf-8")]
    if args.min_gold_words:  # showcase turns that call for a substantive reply (not backchannels)
        def _gold_len(it):
            m = it["messages"][-1]
            return len(m["content"].split()) if m.get("role") == "assistant" else 0
        items = [it for it in items if _gold_len(it) >= args.min_gold_words]
    for i, it in enumerate(items[:args.n], 1):
        npz = np.load(it["npz_path"])
        cross = getattr(mm, "use_crossattn", False)
        alpha = torch.from_numpy(npz["alpha"].astype(np.float32)).unsqueeze(0).to(args.device)
        if args.zero_alpha:
            alpha = torch.zeros_like(alpha)
        mu = torch.from_numpy(npz["mu"].astype(np.float32)).unsqueeze(0).to(args.device)

        def _t(name):
            return torch.from_numpy(npz[name].astype(np.float32)).unsqueeze(0).to(args.device)
        base = (dict(text_emb=_t("text_emb"), audio_seq=_t("audio_seq"), video_seq=_t("video_seq"))
                if cross else dict(audio_feat=_t("audio_feat"), video_feat=_t("video_feat")))

        msgs = it["messages"]
        gold = msgs[-1]["content"] if (msgs and msgs[-1].get("role") == "assistant") else ""
        client = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")  # current client turn
        if len(msgs) > 0 and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]

        text = chat_text(tok, msgs)
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        # variants: sweep a state dim (needs state_proj) / real-vs-zeroed AV / plain
        dim_idx = {"chg": 0, "aro": 1, "val": 2}
        if args.sweep_dim and getattr(mm, "state_proj", None) is not None:
            di = dim_idx[args.sweep_dim]
            variants = []
            for v in [float(x) for x in args.sweep_vals.split(",")]:
                m2 = mu.clone(); m2[0, di] = v
                variants.append((f"{args.sweep_dim}={v:.2f}", dict(base), m2))
        elif args.diag_av:
            bz = dict(base)
            for k in ("audio_feat", "video_feat", "audio_seq", "video_seq"):
                if k in bz:
                    bz[k] = torch.zeros_like(bz[k])
            variants = [("av=real", dict(base), mu), ("av=zero", bz, mu)]
        else:
            variants = [("real-mu", dict(base), mu)]

        print(f"\n=== sample {i} ===")
        print("CLIENT   :", client)
        print("GOLD     :", gold)
        for label, mm_in, state in variants:
            st = state if getattr(mm, "state_proj", None) is not None else None
            full_ids, inputs_embeds, attn2, K = build_inputs_embeds(mm, input_ids, attn, alpha, st, **mm_in)
            torch.manual_seed(args.seed + i)  # SAME seed across variants -> any difference is from the varied input
            out = gen_lm.generate(
                input_ids=full_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attn2,
                max_new_tokens=args.max_new_tokens,
                do_sample=not args.greedy,
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
            gen = out[0, full_ids.size(1):]
            print(f"  [{label}] {tok.decode(gen, skip_special_tokens=True).strip()}")

if __name__ == "__main__":
    main()
