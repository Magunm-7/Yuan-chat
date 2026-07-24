"""
从候选池离线造 DPO 偏好对,reward 项可配 —— 用于逐项消融(behaviour / +rel / +faith / +len)。
behaviour 恒在(baseline);--terms 追加 rel,faith,len 中的若干项。

  # 消融 A: 只 behaviour
  python scripts/make_pairs_offline.py --pool data/annomi/cand_pool.jsonl --terms "" --out data/annomi/pairs_A.jsonl
  # 消融 B: behaviour + relevance
  python scripts/make_pairs_offline.py --pool data/annomi/cand_pool.jsonl --terms rel --out data/annomi/pairs_B.jsonl
"""
from __future__ import annotations
import json, argparse
import numpy as np

BEHAV_REWARD = {"reflection": 1.0, "question": 0.6, "other": 0.2, "therapist_input": -0.5}
REL_MID, REL_SLOPE, W_REL, W_FAB, W_LEN, TARGET_LEN = 0.28, 12.0, 0.4, 1.0, 0.02, 15


def score(c, terms):
    lab = c["behaviour"]; rel = float(c["rel"]); faith = float(c.get("faith") or 0.0)
    words = int(c["words"]); b = BEHAV_REWARD[lab]
    if "rel" in terms:
        gate = 1.0 / (1.0 + np.exp(-(rel - REL_MID) * REL_SLOPE))
        s = b * gate + W_REL * rel
    else:
        s = b
    if "faith" in terms:
        s -= W_FAB * (faith if lab == "reflection" else 0.0)
    if "len" in terms:
        s -= W_LEN * max(0, words - TARGET_LEN)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default="data/annomi/cand_pool.jsonl")
    ap.add_argument("--terms", default="", help="逗号分隔: rel,faith,len (behaviour 恒在)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_gap", type=float, default=0.3)
    args = ap.parse_args()

    terms = set(t.strip() for t in args.terms.split(",") if t.strip())
    rows = [json.loads(l) for l in open(args.pool, encoding="utf-8")]
    pairs, gaps, skip = [], [], 0
    for r in rows:
        cs = r["candidates"]
        scored = sorted(((score(c, terms), c) for c in cs), key=lambda t: -t[0])
        best, worst = scored[0], scored[-1]
        gap = best[0] - worst[0]
        if gap < args.min_gap or best[1]["reply"] == worst[1]["reply"]:
            skip += 1
            continue
        gaps.append(gap)
        pairs.append({"prompt_messages": r["prompt_messages"],
                      "chosen": best[1]["reply"], "rejected": worst[1]["reply"]})
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    cw = np.mean([len(p["chosen"].split()) for p in pairs]) if pairs else 0
    rw = np.mean([len(p["rejected"].split()) for p in pairs]) if pairs else 0
    print(f"terms={terms or '{behaviour}'}  保留 {len(pairs)}/{len(rows)}  "
          f"分差均值 {np.mean(gaps) if gaps else 0:.3f}  chosen词 {cw:.1f} rejected词 {rw:.1f}  -> {args.out}")


if __name__ == "__main__":
    main()
