"""
新旧 DPO 横向对比:SFT / DPO-v1(旧reward) / DPO-v2(新reward=加忠实性+长度惩罚)。
指标加"捏造率"(对 reflection 生成用 faithfulness 打分)和词数,直接验收这轮迭代。

一次加载 base+SFT,挂两个 adapter,disable/set_adapter 切换三配置。

  python scripts/eval_dpo_v2.py --n 60
"""
from __future__ import annotations
import os, json, argparse
from contextlib import nullcontext
import numpy as np
import torch

BEHAV_REWARD = {"reflection": 1.0, "question": 0.6, "other": 0.2, "therapist_input": -0.5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--dpo_v1", default="outputs/dpo/qwen7b_dpo")
    ap.add_argument("--dpo_v2", default="outputs/dpo/qwen7b_dpo_v2")
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--fab_thresh", type=float, default=0.34)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    from faithfulness import score_faithfulness
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[load] base + SFT(merge)")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft if os.path.isdir(sft) else args.sft_lora, adapter_name="sft")
    lm = lm.merge_and_unload()

    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    d = np.load("outputs/evaluator/behaviour_clf.npz", allow_pickle=True)
    B = (d["coef"], d["intercept"], [str(k) for k in d["keys"]])

    configs = [("SFT 单次", None)]
    lm2 = lm
    if os.path.isdir(args.dpo_v1):
        lm2 = PeftModel.from_pretrained(lm, args.dpo_v1, adapter_name="v1")
        configs.append(("DPO-v1 单次", "v1"))
    if os.path.isdir(args.dpo_v2):
        if isinstance(lm2, PeftModel):
            lm2.load_adapter(args.dpo_v2, adapter_name="v2")
        else:
            lm2 = PeftModel.from_pretrained(lm, args.dpo_v2, adapter_name="v2")
        configs.append(("DPO-v2 单次", "v2"))
    lm2.eval()
    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    def behav(texts):
        v = np.asarray(st.encode(texts), dtype=np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return [B[2][i] for i in (v @ B[0].T + B[1]).argmax(1)], v

    @torch.no_grad()
    def run(tag, adapter):
        labs_all, words_all, fabs = [], [], []
        for r in rows:
            msgs = r["messages"][:-1]
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
            except TypeError:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
            if adapter is None and isinstance(lm2, PeftModel):
                ctx = lm2.disable_adapter()
            else:
                if isinstance(lm2, PeftModel):
                    lm2.set_adapter(adapter)
                ctx = nullcontext()
            with ctx:
                o = lm2.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.9, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
            rep = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip()
            rep = rep.split("</think>")[-1].strip()
            lab = behav([rep])[0][0]
            us = [m["content"] for m in msgs if m["role"] == "user"]
            cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
            ctx_text = " ".join([cur] + [m["content"] for m in msgs[:-1]])
            fab = score_faithfulness(rep, ctx_text)[0] if lab == "reflection" else 0.0
            labs_all.append(lab); words_all.append(len(rep.split())); fabs.append(fab)
        n = len(labs_all)
        refl = sum(l == "reflection" for l in labs_all) / n * 100
        other = sum(l == "other" for l in labs_all) / n * 100
        # 捏造率:reflection 里 faith penalty 超阈值的占比(占全部)
        fabrate = sum(f > args.fab_thresh for f in fabs) / n * 100
        print(f"  {tag:14s} reflection={refl:4.1f}%  敷衍={other:4.1f}%  "
              f"词数={np.mean(words_all):4.1f}  捏造率={fabrate:4.1f}%", flush=True)

    print(f"\n=== 新旧 DPO 横向对比({len(rows)} 条 holdout,单次采样)===")
    for tag, ad in configs:
        run(tag, ad)
    print("\n判读:v2 相对 v1 应 捏造率↓ + 词数↓(24→~12),同时 reflection 不塌、敷衍不反弹。")


if __name__ == "__main__":
    main()
