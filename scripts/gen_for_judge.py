"""
为 LLM-judge 生成三阶段回复:base(未微调) / SFT / DPO,对同一批 holdout 各生成一条。
三者喂完全相同的 prompt(holdout messages,含状态标记),保证输入一致、对比公平。
导出 responses_for_judge.jsonl,拿到本地用 gpt4_judge.py 评。

加载技巧:base+SFT-adapter(不 merge)时 disable=base / enable=SFT;
再 merge SFT + 挂 DPO 得 DPO —— 一次加载覆盖三配置。

  python scripts/gen_for_judge.py --n 60 --out data/annomi/responses_for_judge.jsonl
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
    ap.add_argument("--out", default="data/annomi/responses_for_judge.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    @torch.no_grad()
    def gen(model, msgs, ctx=None):
        from contextlib import nullcontext
        try:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                          enable_thinking=False)
        except TypeError:
            txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
        with (ctx or nullcontext()):
            torch.manual_seed(args.seed)
            o = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                               temperature=args.temperature, top_p=0.9, repetition_penalty=1.05,
                               pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip().split("</think>")[-1].strip()

    print("[load] base + SFT adapter(不 merge)")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft if os.path.isdir(sft) else args.sft_lora, adapter_name="sft")
    lm.eval()

    recs = []
    for i, r in enumerate(rows, 1):
        msgs = r["messages"][:-1]
        base_reply = gen(lm, msgs, lm.disable_adapter())   # 关掉 adapter = 未微调 base
        lm.set_adapter("sft")
        sft_reply = gen(lm, msgs)
        us = [m["content"] for m in msgs if m["role"] == "user"]
        cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
        recs.append({"prompt_messages": msgs, "user_text": cur,
                     "gold": r["messages"][-1]["content"],
                     "base": base_reply, "sft": sft_reply})
        if i % 20 == 0:
            print(f"  base/sft {i}/{len(rows)}", flush=True)

    print("[load] merge SFT + 挂 DPO adapter")
    lm = lm.merge_and_unload()
    lm = PeftModel.from_pretrained(lm, args.dpo_lora).eval()
    for i, rec in enumerate(recs, 1):
        rec["dpo"] = gen(lm, rec["prompt_messages"])
        if i % 20 == 0:
            print(f"  dpo {i}/{len(recs)}", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for rec in recs:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"saved {len(recs)} 条(base/sft/dpo/gold) -> {args.out}")


if __name__ == "__main__":
    main()
