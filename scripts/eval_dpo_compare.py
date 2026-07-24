"""
DPO 横向对比:一次加载,四种配置全测。

  SFT + 单次      基线(关掉 BoN 的样子)
  SFT + BoN-8     当前 demo 的质量
  DPO + 单次      ← 核心:应当 ≈ SFT+BoN-8,才说明训练把 reward 吃进了权重
  DPO + BoN-8     上限

技巧:DPO adapter 挂在"base+SFT 已 merge"的权重上,所以 disable_adapter 得到的
正是 SFT 模型 —— 同一份权重覆盖全部四种配置,不必反复加载。

  python scripts/eval_dpo_compare.py --n 60
"""
from __future__ import annotations
import os, re, json, argparse
from contextlib import nullcontext as _null
import numpy as np
import torch

BEHAV_REWARD = {"reflection": 1.0, "question": 0.6, "other": 0.2, "therapist_input": -0.5}
REL_MID, REL_SLOPE, W_REL, W_FAB = 0.28, 12.0, 0.4, 1.2
ANAPHORA_RE = re.compile(
    r"\b(you (said|mentioned|told me|were saying)|we (talked|discussed|spoke)|"
    r"last time|earlier you|as you (said|mentioned))\b", re.I)
UNGROUNDED_RE = re.compile(
    r"\b(referral|referred (by|from)|another (counselor|therapist|doctor)|"
    r"your (doctor|physician|husband|wife|partner|boss|family)|"
    r"(the|your) (appointment|schedule|test results|chart|paperwork)|"
    r"i (noticed|can see|see) (that )?you('re| are)|"
    r"\d+\s*(days?|weeks?|months?|years?|drinks?|cigarettes?|pounds?|kilos?|kilograms?))\b", re.I)


def _norm(v):
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


class Scorer:
    def __init__(self, device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        d = np.load("outputs/evaluator/behaviour_clf.npz", allow_pickle=True)
        self.coef, self.intercept = d["coef"], d["intercept"]
        self.keys = [str(k) for k in d["keys"]]

    def __call__(self, user_text, replies, ctx):
        ctx = [c for c in ([user_text] + list(ctx)) if c.strip()]
        v = _norm(np.asarray(self.st.encode([user_text] + replies + ctx), dtype=np.float32))
        n = len(replies)
        v_user, v_rep, v_ctx = v[0], v[1:1 + n], v[1 + n:]
        rel = v_rep @ v_user
        labs = [self.keys[i] for i in (v_rep @ self.coef.T + self.intercept).argmax(1)]
        out = []
        for i, (r, lab) in enumerate(zip(replies, labs)):
            fab = 0.0
            if ANAPHORA_RE.search(r) or UNGROUNDED_RE.search(r):
                sim = float((v_ctx @ v_rep[i]).max()) if len(v_ctx) else 0.0
                fab = float(np.clip((0.5 - sim) / 0.5, 0.0, 1.0))
            gate = 1.0 / (1.0 + np.exp(-(float(rel[i]) - REL_MID) * REL_SLOPE))
            total = BEHAV_REWARD[lab] * gate + W_REL * float(rel[i]) - W_FAB * fab
            out.append({"reply": r, "behaviour": lab, "rel": float(rel[i]),
                        "fab": fab, "score": float(total)})
        return sorted(out, key=lambda d: -d["score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--dpo_lora", default="outputs/dpo/qwen7b_dpo")
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temp_single", type=float, default=0.6)
    ap.add_argument("--temp_bon", type=float, default=0.9)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[load] base + SFT(merge)")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft_dir = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft_dir if os.path.isdir(sft_dir) else args.sft_lora)
    lm = lm.merge_and_unload()
    has_dpo = os.path.isdir(args.dpo_lora)
    if has_dpo:
        print("[load] + DPO adapter")
        lm = PeftModel.from_pretrained(lm, args.dpo_lora)
    lm.eval()
    scorer = Scorer("cpu")

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    @torch.no_grad()
    def run(tag, use_dpo, k):
        tot, dist, wl = [], {}, []
        for r in rows:
            msgs = r["messages"][:-1]
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(txt, return_tensors="pt", truncation=True,
                      max_length=1536).input_ids.to(dev)
            ctx = lm.disable_adapter() if (has_dpo and not use_dpo) else _null()
            with ctx:
                out = lm.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                  temperature=args.temp_bon if k > 1 else args.temp_single,
                                  top_p=0.95 if k > 1 else 0.9, repetition_penalty=1.05,
                                  num_return_sequences=k, pad_token_id=tok.eos_token_id)
            cands = []
            for o in out:
                c = tok.decode(o[ids.shape[1]:], skip_special_tokens=True).strip()
                c = c.split("</think>")[-1].strip()
                if c and c not in cands:
                    cands.append(c)
            if not cands:
                continue
            us = [m["content"] for m in msgs if m["role"] == "user"]
            cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
            ranked = scorer(cur, cands, [m["content"] for m in msgs[:-1]])
            best = ranked[0]
            tot.append(best["score"]); wl.append(len(best["reply"].split()))
            dist[best["behaviour"]] = dist.get(best["behaviour"], 0) + 1
        n = max(1, len(tot))
        pct = {k2: dist.get(k2, 0) / n * 100 for k2 in BEHAV_REWARD}
        print(f"  {tag:16s} 平均reward={np.mean(tot):+.3f}  " +
              "  ".join(f"{k2[:6]}={pct[k2]:4.1f}%" for k2 in BEHAV_REWARD) +
              f"  词数={np.mean(wl):.1f}", flush=True)
        return float(np.mean(tot))

    print(f"\n=== 横向对比({len(rows)} 条 holdout)===")
    s1 = run("SFT + 单次", False, 1)
    s8 = run("SFT + BoN-8", False, 8)
    if has_dpo:
        d1 = run("DPO + 单次", True, 1)
        d8 = run("DPO + BoN-8", True, 8)
        print(f"\n判据:DPO单次({d1:+.3f}) vs SFT+BoN8({s8:+.3f}) -> "
              f"{'训练有效,可关掉 BoN' if d1 >= s8 - 0.02 else '未达 BoN 水平,建议继续用 BoN'}")
        print(f"      DPO单次({d1:+.3f}) vs SFT单次({s1:+.3f}) -> 提升 {d1-s1:+.3f}")
    else:
        print("\n(未找到 DPO adapter,仅跑了 SFT 两档)")


if __name__ == "__main__":
    main()
