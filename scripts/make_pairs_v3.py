# -*- coding: utf-8 -*-
"""v3 DPO 偏好对: 纯离散 reward, question 用 oc_clf 拆 open/closed。CPU(需MiniLM)。
reward = {reflection:1.0, q_open:1.0, q_closed:0.1, other:0.2, therapist_input:-0.5}
  python make_pairs_v3.py --pool cand_pool_v3.jsonl --out pairs_v3.jsonl"""
import json, argparse, numpy as np
from collections import Counter

REWARD = {"reflection":1.0, "q_open":1.0, "q_closed":0.1, "other":0.2, "therapist_input":-0.5}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/annomi/cand_pool_v3.jsonl")
    ap.add_argument("--oc_clf", default="outputs/evaluator/oc_clf.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_gap", type=float, default=0.3)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    # 收集所有 question 候选的 reply, 一次性嵌入+分类 open/closed
    d = np.load(args.oc_clf, allow_pickle=True)
    coef, intercept, keys = d["coef"], d["intercept"], [str(k) for k in d["keys"]]
    q_texts, q_ref = [], []
    for ri, r in enumerate(rows):
        for ci, c in enumerate(r["candidates"]):
            if c.get("behaviour")=="question":
                q_texts.append(c["reply"]); q_ref.append((ri,ci))
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    if q_texts:
        E = np.asarray(m.encode(q_texts, batch_size=256, show_progress_bar=False), dtype=np.float32)
        logits = E @ coef.T + intercept
        pred = logits.argmax(1)
        oc = {}
        for (ri,ci), p in zip(q_ref, pred): oc[(ri,ci)] = keys[p]   # 'open'/'closed'
    else:
        oc = {}

    def key_of(ri, ci, c):
        b = c.get("behaviour")
        if b=="question":
            return "q_open" if oc.get((ri,ci))=="open" else "q_closed"
        return b
    def score(ri, ci, c): return REWARD.get(key_of(ri,ci,c), 0.0)

    pairs=[]; ch_keys=Counter(); rj_keys=Counter(); gaps=[]
    for ri, r in enumerate(rows):
        cs=r["candidates"]
        if len(cs)<2: continue
        scored=sorted(((score(ri,ci,c), ci, c) for ci,c in enumerate(cs)), key=lambda t:-t[0])
        (bs,bci,bc),(ws,wci,wc)=scored[0],scored[-1]
        if bs-ws<args.min_gap or bc["reply"]==wc["reply"]: continue
        gaps.append(bs-ws)
        ch_keys[key_of(ri,bci,bc)]+=1; rj_keys[key_of(ri,wci,wc)]+=1
        pairs.append({"prompt_messages":r["prompt_messages"],"chosen":bc["reply"],"rejected":wc["reply"]})
    with open(args.out,"w",encoding="utf-8") as f:
        for p in pairs: f.write(json.dumps(p,ensure_ascii=False)+"\n")

    print(f"保留 {len(pairs)}/{len(rows)} 对  分差均值 {np.mean(gaps) if gaps else 0:.3f}")
    print(f"chosen  分布: {dict(ch_keys.most_common())}")
    print(f"rejected分布: {dict(rj_keys.most_common())}")
    cw=np.mean([len(p['chosen'].split()) for p in pairs]) if pairs else 0
    rw=np.mean([len(p['rejected'].split()) for p in pairs]) if pairs else 0
    print(f"chosen词 {cw:.1f}  rejected词 {rw:.1f}  -> {args.out}")
    print("\n样例(前6对):")
    for p in pairs[:6]:
        print(f"  C: {p['chosen'][:80]}")
        print(f"  R: {p['rejected'][:80]}\n")

if __name__ == "__main__":
    main()
