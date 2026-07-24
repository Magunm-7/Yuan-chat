"""
采样候选池并保存全部分项(reply/behaviour/rel/faith/words)。
采一次, 之后任意 reward 消融都离线复用这批候选, 不再重复采样。

  python scripts/sample_candidates.py --n 300 --k 8 --out data/annomi/cand_pool.jsonl
"""
from __future__ import annotations
import json, argparse, urllib.request
import numpy as np

API = "http://localhost:8000/generate"


def gen(messages, k, temperature):
    body = json.dumps({"messages": messages, "n": k, "temperature": temperature,
                       "max_new_tokens": 96}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/train.jsonl")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--out", default="data/annomi/cand_pool.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")]
    idx = np.random.default_rng(args.seed).permutation(len(rows))[:args.n]
    keep = ("reply", "behaviour", "rel", "faith", "words")
    used = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for c, i in enumerate(idx, 1):
            msgs = rows[i]["messages"][:-1]
            us = [m["content"] for m in msgs if m.get("role") == "user"]
            cur = us[-1].split("]" + chr(10), 1)[-1].strip() if us else ""
            if len(cur.split()) < 3:
                continue
            try:
                d = gen(msgs, args.k, args.temperature)
            except Exception:
                continue
            cands = d.get("candidates", [])
            if len(cands) < 2:
                continue
            slim = [{k: cc.get(k) for k in keep} for cc in cands]
            f.write(json.dumps({"prompt_messages": msgs, "candidates": slim},
                               ensure_ascii=False) + "\n")
            used += 1
            if c % 50 == 0:
                print(f"  {c}/{len(idx)}  有效 {used}", flush=True)
    print(f"saved {used} 条候选池 -> {args.out}")


if __name__ == "__main__":
    main()
