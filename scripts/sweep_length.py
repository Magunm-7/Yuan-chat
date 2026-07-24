"""
标定长度惩罚权重 W_len。一次采样,离线扫多个 W_len(在已采候选上重排,不重新生成)。

核心是看两条曲线随 W_len 的变化,判断有没有甜点区:
  - 忠实度↑(top1 的子句 min-relevance,越高越不捏造 —— 因为捏造依附长回复,压长度→淘汰长捏造子句)
  - 敷衍率不反弹(top1 是 other 语气词的比例;长度惩罚偏爱短回复,最短的就是敷衍,需警惕)

对 80 条 holdout 各采 8 候选(走 demo /generate),记录 words + 子句min-rel,再离线扫。

  python scripts/sweep_length.py --n 80
"""
from __future__ import annotations
import re, json, argparse, urllib.request
import numpy as np

API = "http://localhost:8000/generate"


def split_clauses(text):
    parts = re.split(r"[.?!;,]|\band\b|\bbut\b|\bso\b", text, flags=re.I)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def gen(messages, k, temperature):
    body = json.dumps({"messages": messages, "n": k, "temperature": temperature,
                       "max_new_tokens": 96}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--target_len", type=int, default=15)
    ap.add_argument("--fab_thresh", type=float, default=0.25, help="子句min-rel 低于此算含捏造(参考指标)")
    args = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def norm(xs):
        v = np.asarray(st.encode(xs), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]
    data, used = [], 0
    for i, r in enumerate(rows, 1):
        msgs = r["messages"][:-1]
        try:
            d = gen(msgs, args.k, args.temperature)
        except Exception:
            continue
        cands = d.get("candidates", [])
        cur = d.get("user_text", "")
        if len(cands) < 2 or len(cur.split()) < 3:
            continue
        cvec = norm([cur])[0]
        for c in cands:
            c["words"] = len(c["reply"].split())
            cl = split_clauses(c["reply"]) or [c["reply"]]
            c["clause_min_rel"] = float((norm(cl) @ cvec).min())
        data.append(cands)
        used += 1
        if i % 20 == 0:
            print(f"  采样 {i}/{len(rows)}  有效 {used}", flush=True)

    print(f"\n有效输入 {used} 条,共 {sum(len(c) for c in data)} 候选\n")
    print(f"{'W_len':>6} {'top1词数':>8} {'敷衍率':>7} {'忠实度':>7} {'捏造率':>7}  (target_len={args.target_len})")
    print("-" * 52)
    for wlen in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3]:
        top1 = []
        for cands in data:
            best = max(cands, key=lambda c: c["score"] - wlen * max(0, c["words"] - args.target_len))
            top1.append(best)
        aw = np.mean([t["words"] for t in top1])
        other = np.mean([t["behaviour"] == "other" for t in top1]) * 100
        faith = np.mean([t["clause_min_rel"] for t in top1])
        fabr = np.mean([t["clause_min_rel"] < args.fab_thresh for t in top1]) * 100
        print(f"{wlen:>6.2f} {aw:>8.1f} {other:>6.1f}% {faith:>7.3f} {fabr:>6.1f}%")
    print("\n判读:忠实度↑ / 捏造率↓ / 敷衍率不反弹 的最大 W_len = 甜点。gold 词数≈11.6。")


if __name__ == "__main__":
    main()
