# -*- coding: utf-8 -*-
"""在 100 条 holdout 上生成 base / v2 / v3 三方回复(gold 取自 holdout), 供 judge。
推理设置与 eval_bcw/gen_for_judge 一致(temp0.6/seed0/top_p0.9, 纯文本)。"""
import os, json, argparse, torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3-14B")
    ap.add_argument("--v2", default="outputs/mm_sft/qwen14b_sft_2048_bcw2/lora_adapter")
    ap.add_argument("--v3", default="outputs/mm_sft/qwen14b_sft_2048_bcw3/lora_adapter")
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--out", default="data/annomi/responses_judge100.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    @torch.no_grad()
    def gen(model, msgs):
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
        torch.manual_seed(args.seed)
        o = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                           temperature=args.temperature, top_p=0.9, repetition_penalty=1.05,
                           pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip().split("</think>")[-1].strip()

    def gen_all(model, tag):
        outs = []
        for i, r in enumerate(rows):
            outs.append(gen(model, r["messages"][:-1]))
            if (i+1) % 25 == 0: print(f"  [{tag}] {i+1}/{len(rows)}", flush=True)
        return outs

    print("[load] base")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev).eval()
    base_out = gen_all(lm, "base")

    print("[load] v2 adapter")
    lm = PeftModel.from_pretrained(lm, args.v2, adapter_name="v2").eval()
    v2_out = gen_all(lm, "v2")

    print("[load] v3 adapter")
    lm.load_adapter(args.v3, adapter_name="v3")
    lm.set_adapter("v3")
    v3_out = gen_all(lm, "v3")

    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({
                "prompt_messages": r["messages"][:-1],
                "user_text": r["messages"][-2]["content"],
                "gold": r["messages"][-1]["content"],
                "base": base_out[i], "v2": v2_out[i], "v3": v3_out[i],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} -> {args.out}")

if __name__ == "__main__":
    main()
