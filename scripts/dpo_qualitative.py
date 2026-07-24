"""
DPO 定性对比:同几条 holdout 输入,SFT 单次 vs DPO 单次 vs gold 并排。
纯 reward 数字无法区分"变长=更丰富"还是"变长=注水凑长度",必须人工读文本。
disable_adapter() 得到 SFT,同一份权重两用。

  python scripts/dpo_qualitative.py --n 10
"""
from __future__ import annotations
import os, json, argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--dpo_lora", default="outputs/dpo/qwen7b_dpo")
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft_dir = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft_dir if os.path.isdir(sft_dir) else args.sft_lora)
    lm = lm.merge_and_unload()
    lm = PeftModel.from_pretrained(lm, args.dpo_lora).eval()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    @torch.no_grad()
    def gen(msgs, use_dpo):
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
        from contextlib import nullcontext
        ctx = nullcontext() if use_dpo else lm.disable_adapter()
        torch.manual_seed(args.seed)
        with ctx:
            o = lm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                            temperature=args.temperature, top_p=0.9, repetition_penalty=1.05,
                            pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()

    for r in rows:
        msgs = r["messages"][:-1]
        gold = r["messages"][-1]["content"]
        us = [m["content"] for m in msgs if m["role"] == "user"]
        cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
        print(f"\n{'='*80}\n来访者: {cur[:150]}")
        print(f"  GOLD ({len(gold.split()):2d}词): {gold[:160]}")
        s = gen(msgs, False); print(f"  SFT  ({len(s.split()):2d}词): {s[:160]}")
        d = gen(msgs, True);  print(f"  DPO  ({len(d.split()):2d}词): {d[:160]}")


if __name__ == "__main__":
    main()
