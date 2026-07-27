# -*- coding: utf-8 -*-
"""评估 backchannel 降权后的新 14B SFT: 在同一批 holdout 上生成, 对比旧 SFT 的坍缩。
推理路径与 gen_for_judge 完全一致(base + SFT LoRA, temp0.6/seed0/top_p0.9, 纯文本)。"""
import os, json, argparse
import numpy as np
import torch

def nw(s): return len((s or "").split())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen3-14B")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen14b_sft_2048_bcw")
    ap.add_argument("--old", default="data/annomi/responses_14b.jsonl")   # 旧 base/sft/dpo/gold
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--out", default="data/annomi/responses_14b_bcw.jsonl")
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
    old = [json.loads(l) for l in open(args.old, encoding="utf-8")][:args.n]

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

    print("[load] base + new SFT-bcw adapter")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft if os.path.isdir(sft) else args.sft_lora)
    lm.eval()

    new_sft = []
    for i, r in enumerate(rows):
        msgs = r["messages"][:-1]
        new_sft.append(gen(lm, msgs))
        if (i+1) % 15 == 0: print(f"  gen {i+1}/{len(rows)}", flush=True)

    # 保存(带旧 sft 便于并排)
    with open(args.out, "w", encoding="utf-8") as f:
        for i, r in enumerate(rows):
            f.write(json.dumps({"user_text": old[i].get("user_text",""),
                                "gold": old[i].get("gold",""),
                                "sft_old": old[i].get("sft",""),
                                "sft_bcw": new_sft[i]}, ensure_ascii=False) + "\n")

    # 行为分布(用现有 behaviour 分类器)
    try:
        import behaviour_scorer as BS
        def dist(texts):
            labs, _ = BS.predict(texts, "cpu")
            n = len(labs)
            return {k: 100*sum(1 for l in labs if l==k)/n for k in BS.KEYS}
        have_beh = True
    except Exception as e:
        print("行为分类器不可用, 跳过:", e); have_beh = False

    old_sft = [old[i].get("sft","") for i in range(len(rows))]
    print("\n" + "="*64)
    print("SFT 坍缩对比 (60 条 holdout, 同 prompt/temp)")
    print("="*64)
    for name, texts in (("旧 SFT", old_sft), ("新 SFT-bcw", new_sft)):
        wl = np.mean([nw(t) for t in texts])
        bc = 100*np.mean([nw(t)<=3 for t in texts])
        line = f"  {name:11s} 平均词长 {wl:5.1f}   坍缩(<=3词) {bc:4.0f}%"
        if have_beh:
            d = dist(texts)
            line += "   " + " ".join(f"{k[:4]}={d[k]:4.0f}%" for k in BS.KEYS)
        print(line)

    print("\n并排样例(旧坍缩的看新的救没救回来):")
    shown=0
    for i in range(len(rows)):
        if nw(old_sft[i])<=3 and nw(new_sft[i])>3:
            print(f"  IN  : {old[i].get('user_text','')[:70]}")
            print(f"  gold: {old[i].get('gold','')[:70]}")
            print(f"  旧SFT: {old_sft[i][:50]!r}   新SFT: {new_sft[i][:80]!r}")
            shown+=1
            if shown>=4: break
    print(f"\nwrote -> {args.out}")

if __name__ == "__main__":
    main()
