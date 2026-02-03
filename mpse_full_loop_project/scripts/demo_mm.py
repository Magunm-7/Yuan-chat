import json, argparse, numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mpse_mvp.mm.train_mm_sft import load_mm_prefix

def chat_text(tok, messages):
    # 用 chat template（如果有）
    if hasattr(tok, "apply_chat_template"):
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # fallback
    s=[]
    for m in messages:
        s.append(f"{m['role'].upper()}: {m['content']}".strip())
    return "\n".join(s) + "\nASSISTANT:"

@torch.no_grad()
def build_inputs_embeds(mm, input_ids, attention_mask, audio_feat, video_feat, alpha):
    B = input_ids.size(0)
    emb = mm.lm.get_input_embeddings()(input_ids)

    a_tok = mm.audio_proj(audio_feat)   # (B,Ka,D)
    v_tok = mm.video_proj(video_feat)   # (B,Kv,D)

    if mm.use_alpha_gate and alpha is not None:
        a_tok = a_tok * alpha[:,0].view(B,1,1)
        v_tok = v_tok * alpha[:,1].view(B,1,1)

    prefix = torch.cat([a_tok, v_tok], dim=1)  # (B,K,D)
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
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    lm = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        torch_dtype=torch.float16 if args.device.startswith("cuda") else torch.float32
    ).to(args.device).eval()

    mm = load_mm_prefix(lm, args.mm_prefix, device=args.device)

    items = [json.loads(l) for l in open(args.index_jsonl, encoding="utf-8")]
    for it in items[:5]:  # demo：先跑前5条
        npz = np.load(it["npz_path"])
        audio = torch.from_numpy(npz["audio_feat"]).unsqueeze(0).to(args.device)
        video = torch.from_numpy(npz["video_feat"]).unsqueeze(0).to(args.device)
        alpha = torch.from_numpy(npz["alpha"]).unsqueeze(0).to(args.device)

        msgs = it["messages"]
        # 如果最后一条是 assistant（训练标签），demo 时去掉，让模型生成
        if len(msgs) > 0 and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]

        text = chat_text(tok, msgs)
        enc = tok(text, return_tensors="pt")
        input_ids = enc["input_ids"].to(args.device)
        attn = enc["attention_mask"].to(args.device)

        full_ids, inputs_embeds, attn2, K = build_inputs_embeds(mm, input_ids, attn, audio, video, alpha)

        out = lm.generate(
            input_ids=full_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attn2,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
        gen = out[0, full_ids.size(1):]  # 只取新生成部分
        print("\n=== TURN", it.get("meta", {}).get("turn_id"), "===")
        print(tok.decode(gen, skip_special_tokens=True))

if __name__ == "__main__":
    main()
