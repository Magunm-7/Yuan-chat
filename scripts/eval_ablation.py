"""
消融评估:对比多个小 DPO(不同 reward 项)在各自目标指标上的表现。
一次加载 base+SFT,挂多个 adapter,set_adapter 切换。

指标:
  答非所问率  回复与来访者输入的 relevance(MiniLM cos)低于阈值的比例(relevance 的正面目标)
  复读率     回复与来访者输入的 bigram 重叠(relevance 的副作用: 外推成复读机)
  reflection/敷衍  行为分布(确认没把 behaviour 维度带坏)
  词数

  python scripts/eval_ablation.py --adapters A:outputs/dpo/dpo_relA B:outputs/dpo/dpo_relB --n 60
"""
from __future__ import annotations
import os, re, json, argparse
from contextlib import nullcontext
import numpy as np
import torch


def bigram_overlap(a, b):
    def bg(s):
        w = re.findall(r"[a-z']+", s.lower())
        return set(zip(w, w[1:]))
    A, B = bg(a), bg(b)
    return len(A & B) / max(1, len(A))          # 回复里有多少 bigram 抄自输入


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--adapters", nargs="+", default=[], help="tag:path ...")
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--offtopic_thresh", type=float, default=0.2)
    args = ap.parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    from sentence_transformers import SentenceTransformer
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    print("[load] base + SFT(merge)")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft if os.path.isdir(sft) else args.sft_lora, adapter_name="sft")
    lm = lm.merge_and_unload()

    taps = [a.split(":", 1) for a in args.adapters]
    lm2 = lm
    for i, (tag, path) in enumerate(taps):
        if i == 0:
            lm2 = PeftModel.from_pretrained(lm, path, adapter_name=tag)
        else:
            lm2.load_adapter(path, adapter_name=tag)
    lm2.eval()

    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    d = np.load("outputs/evaluator/behaviour_clf.npz", allow_pickle=True)
    B = (d["coef"], d["intercept"], [str(k) for k in d["keys"]])

    def behav(t):
        v = np.asarray(st.encode([t]), dtype=np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return B[2][int((v @ B[0].T + B[1]).argmax(1)[0])]

    def relev(reply, user):
        v = np.asarray(st.encode([reply, user]), dtype=np.float32)
        v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)
        return float(v[0] @ v[1])

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]

    @torch.no_grad()
    def run(tag, adapter):
        offt, rep, refl, other, wl = [], [], [], [], []
        for r in rows:
            msgs = r["messages"][:-1]
            us = [m["content"] for m in msgs if m["role"] == "user"]
            cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
            try:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                              enable_thinking=False)
            except TypeError:
                txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
            ctx = nullcontext() if adapter else (lm2.disable_adapter() if isinstance(lm2, PeftModel) else nullcontext())
            if adapter and isinstance(lm2, PeftModel):
                lm2.set_adapter(adapter)
            with ctx:
                o = lm2.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                 temperature=args.temperature, top_p=0.9, repetition_penalty=1.05,
                                 pad_token_id=tok.eos_token_id)
            rp = tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip().split("</think>")[-1].strip()
            lab = behav(rp)
            offt.append(relev(rp, cur) < args.offtopic_thresh)
            rep.append(bigram_overlap(rp, cur))
            refl.append(lab == "reflection"); other.append(lab == "other"); wl.append(len(rp.split()))
        n = len(offt)
        print(f"  {tag:20s} 答非所问={np.mean(offt)*100:4.1f}%  复读率={np.mean(rep)*100:4.1f}%  "
              f"reflection={np.mean(refl)*100:4.1f}%  敷衍={np.mean(other)*100:4.1f}%  词数={np.mean(wl):4.1f}",
              flush=True)

    print(f"\n=== 消融对比({len(rows)} 条 holdout,单次采样)===")
    run("SFT(基线)", None)
    for tag, _ in taps:
        run(tag, tag)


if __name__ == "__main__":
    main()
