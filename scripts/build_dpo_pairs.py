"""
用 best-of-n 批量构造 DPO 偏好对(rejection sampling,Llama-2 同款做法)。

输入:训练集 prompt(砍掉 gold 回复) —— 纯 on-policy,**不把 gold 放进候选池**,
      否则 chosen 基本都会变成 gold,DPO 退化成变相 SFT、加剧模板化。
过程:每条 prompt 高温采样 n 个候选 -> 复合 reward 打分 -> 最高分=chosen、最低分=rejected
输出:{prompt_messages, chosen, rejected, score_chosen, score_rejected}

通过 HTTP 调 demo 服务的 /generate,复用其已加载模型,不额外占显存。

  python scripts/build_dpo_pairs.py --n_prompts 300 --k 8 --out data/annomi/dpo_pairs.jsonl
"""
from __future__ import annotations
import json, time, argparse, urllib.request
import numpy as np

API = "http://localhost:8000/generate"


def gen(messages, k, temperature, max_new_tokens):
    body = json.dumps({"messages": messages, "n": k, "temperature": temperature,
                       "max_new_tokens": max_new_tokens}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/annomi/mm_sft_final/train.jsonl")
    ap.add_argument("--out", default="data/annomi/dpo_pairs.jsonl")
    ap.add_argument("--n_prompts", type=int, default=300)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--min_gap", type=float, default=0.3, help="分差太小=没有学习信号,丢弃")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.src, encoding="utf-8")]
    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(rows))[:args.n_prompts]
    print(f"源 {len(rows)} 条 -> 采样 {len(idx)} 条 prompt,每条 {args.k} 个候选")

    pairs, skipped = [], {"few_cands": 0, "small_gap": 0, "neg_best": 0,
                          "short_input": 0, "error": 0}
    gaps, t0 = [], time.time()
    for c, i in enumerate(idx, 1):
        msgs = rows[i]["messages"][:-1]          # 砍掉 gold,只留 prompt
        # AnnoMI 的转写切分会产生空/残缺的来访者话语,这类 prompt 只会得到断句片段,
        # 构成的偏好对纯属噪声 —— 在采样前就跳过,省算力
        us = [m["content"] for m in msgs if m.get("role") == "user"]
        cur_txt = us[-1].split("]" + chr(10), 1)[-1].strip() if us else ""
        if len(cur_txt.split()) < 3:
            skipped["short_input"] += 1
            continue
        try:
            d = gen(msgs, args.k, args.temperature, args.max_new_tokens)
        except Exception:
            skipped["error"] += 1
            continue
        cands = d.get("candidates", [])
        if len(cands) < 2:
            skipped["few_cands"] += 1
            continue
        best, worst = cands[0], cands[-1]
        gap = best["score"] - worst["score"]
        if best["score"] < 0:
            skipped["neg_best"] += 1
            continue
        if gap < args.min_gap:
            skipped["small_gap"] += 1
            continue
        gaps.append(gap)
        pairs.append({"prompt_messages": msgs,
                      "chosen": best["reply"], "rejected": worst["reply"],
                      "score_chosen": best["score"], "score_rejected": worst["score"],
                      "behaviour_chosen": best["behaviour"], "behaviour_rejected": worst["behaviour"]})
        if c % 50 == 0:
            el = time.time() - t0
            print(f"  {c}/{len(idx)}  保留 {len(pairs)}  用时 {el/60:.1f} min "
                  f"(~{el/c:.2f}s/条)", flush=True)

    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n保留 {len(pairs)}/{len(idx)} ({len(pairs)/max(1,len(idx))*100:.0f}%)  -> {args.out}")
    print(f"丢弃原因: {skipped}")
    if gaps:
        g = np.array(gaps)
        print(f"分差: mean={g.mean():.3f} median={np.median(g):.3f} p90={np.percentile(g,90):.3f}")
    if pairs:
        import collections
        bc = collections.Counter(p["behaviour_chosen"] for p in pairs)
        br = collections.Counter(p["behaviour_rejected"] for p in pairs)
        print(f"chosen 行为分布  : {dict(bc)}")
        print(f"rejected 行为分布: {dict(br)}")
        print("\n=== 样例 ===")
        for p in pairs[:3]:
            u = [m['content'] for m in p['prompt_messages'] if m['role'] == 'user'][-1]
            print(f"\n-- 来访者: {u.split(']')[-1].strip()[:90]}")
            print(f"   chosen  ({p['score_chosen']:+.2f} {p['behaviour_chosen']}): {p['chosen'][:100]}")
            print(f"   rejected({p['score_rejected']:+.2f} {p['behaviour_rejected']}): {p['rejected'][:100]}")


if __name__ == "__main__":
    main()
