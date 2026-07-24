"""
诊断 100 条 gold 的 31% 误伤:是"真误伤"还是"真人合法引入新信息"?
关键假设:捏造几乎只在 reflection 时发生 -> 若只对 reflection 生效,误伤率应大降。

  按 behaviour 拆分 gold 误伤率 + 打印被 flag 的 gold 样例(人工判性质)。
"""
from __future__ import annotations
import json, argparse
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=0.45)
    args = ap.parse_args()

    from faithfulness import score_faithfulness
    from behaviour_scorer import predict

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]
    recs = []
    for r in rows:
        gold = r["messages"][-1]["content"]
        us = [m["content"] for m in r["messages"][:-1] if m["role"] == "user"]
        cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
        hist = " ".join(m["content"] for m in r["messages"][:-1])
        pen, ung = score_faithfulness(gold, cur + " " + hist, args.thresh)
        recs.append({"gold": gold, "cur": cur, "pen": pen, "ung": ung})

    labs, _ = predict([x["gold"] for x in recs], "cpu")
    for x, l in zip(recs, labs):
        x["behav"] = l

    print("=== 按 behaviour 拆分 gold 误伤率 ===")
    for b in ["reflection", "question", "therapist_input", "other"]:
        sub = [x for x in recs if x["behav"] == b]
        if not sub:
            continue
        flag = np.mean([x["pen"] > 0 for x in sub]) * 100
        mp = np.mean([x["pen"] for x in sub])
        print(f"  {b:16s} n={len(sub):3d}  flag率={flag:4.0f}%  平均penalty={mp:.3f}")

    print("\n=== 被 flag 的 gold 样例(判断:真捏造 vs 合法引入)===")
    shown = 0
    for x in recs:
        if x["pen"] > 0.5 and shown < 12:
            shown += 1
            print(f"  [{x['behav'][:6]}] 未grounded={x['ung']}")
            print(f"     来访者: {x['cur'][:80]}")
            print(f"     gold  : {x['gold'][:110]}")


if __name__ == "__main__":
    main()
