"""
实体级忠实性打分(reward 的 fabrication 项,替换早先"太 demo"的关键词正则)。

捏造 = 回复引入了上下文没有的**具体实体**。相对语义相似度的优势:只盯具体名词/数字,
天然跳过情感/态度推断(drunk/okay/concerned 不进实体表),不误伤合理的复杂反映。

打磨点(相对烟雾测试):
  1. 去功能词碎片:bit/lot/couple 这类量词性名词被 spaCy 误标为 NOUN -> 用黑名单挡掉
  2. 数字单独处理:抓"三天一周"这种频率断言(名词 day/week 与上下文 weekend 相似会漏,
     真正编的是数字 three)

  score_faithfulness(reply, context) -> (penalty in [0,1], 未grounded实体列表)

  python scripts/faithfulness.py            # 6 条回归 + 100 条 gold 误伤率
"""
from __future__ import annotations
import argparse
import numpy as np

# 被 spaCy 标成 NOUN 但其实是量词/抽象碎片,不是可捏造的具体实体
FRAGMENTS = {
    "bit", "lot", "couple", "thing", "things", "way", "ways", "kind", "kinds",
    "sort", "sorts", "part", "parts", "one", "ones", "bunch", "deal", "none",
    "stuff", "point", "side", "case", "fact", "matter", "number", "amount",
    "something", "anything", "everything", "nothing", "someone", "everyone",
}
_M = {}
CONC_PATH = "/root/autodl-tmp/concreteness.txt"   # Brysbaert et al. 2014, 40k 词具体性评分
CONC_THRESH = 4.0                                 # >=4.0 视为"具体实体"(bus 4.9 vs stress 2.8)


def _load():
    if "nlp" not in _M:
        import spacy
        from sentence_transformers import SentenceTransformer
        _M["nlp"] = spacy.load("en_core_web_sm")
        _M["st"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        conc = {}
        try:
            with open(CONC_PATH, encoding="utf-8", errors="ignore") as f:
                next(f)
                for line in f:
                    p = line.split("\t")
                    if len(p) >= 3:
                        try:
                            conc[p[0].strip().lower()] = float(p[2])
                        except ValueError:
                            pass
        except FileNotFoundError:
            pass
        _M["conc"] = conc
    return _M["nlp"], _M["st"]


def _norm(st, xs):
    v = np.asarray(st.encode(xs), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def _parse(nlp, text):
    doc = nlp(text)
    conc = _M.get("conc", {})
    nouns, nums = [], []
    for t in doc:
        if t.like_num or t.pos_ == "NUM":
            nums.append(t.lemma_.lower())
        elif t.pos_ in ("NOUN", "PROPN") and not t.is_stop and len(t.lemma_) > 1 \
                and t.lemma_.lower() not in FRAGMENTS:
            lm = t.lemma_.lower()
            c = conc.get(lm)
            # 只查具体实体:专有名词(可能是捏造的人名/地名)总查;抽象概念名词(stress/confidence)豁免
            if t.pos_ == "PROPN" or (c is not None and c >= CONC_THRESH):
                nouns.append(lm)
    return list(dict.fromkeys(nouns)), list(dict.fromkeys(nums))


def score_faithfulness(reply, context, thresh=0.45):
    """penalty in [0,1] = 未grounded具体实体占比;越高越可能捏造。"""
    nlp, st = _load()
    r_nouns, r_nums = _parse(nlp, reply)
    c_nouns, c_nums = _parse(nlp, context)
    items = r_nouns + r_nums
    if not items:
        return 0.0, []
    ung = []
    # 名词:与上下文名词的最大语义相似度
    if r_nouns:
        if c_nouns:
            sims = _norm(st, r_nouns) @ _norm(st, c_nouns).T
            for i, e in enumerate(r_nouns):
                if float(sims[i].max()) < thresh:
                    ung.append(e)
        else:
            ung.extend(r_nouns)
    # 数字:上下文完全没有数字而回复有具体数字 = 编造的断言
    for n in r_nums:
        if n not in c_nums and not c_nums:
            ung.append(n)
    return len(ung) / len(items), ung


CASES = [
    ("捏造(跳公交)", "I was at a pub crawl, you have a few drinks.",
     "So you were doing a pub crawl and you jumped off the stairs of the bus and landed on your ankle?"),
    ("捏造数字(三天一周)", "me and my friends party on the weekend, just have a few drinks.",
     "So it sounds like you are drinking about three days a week."),
    ("半真半假(从学校回家)", "I can drink seven or eight drinks if I am going hard.",
     "So you are drinking seven to eight drinks and a couple more when you get home from school."),
    ("忠实推进(情感延伸)", "Well, it is normal.", "It is normal? So it is okay to be a little bit drunk?"),
    ("忠实复述", "me and my friends party on the weekend, just have a few drinks.",
     "So a few drinks is nothing new to you?"),
    ("答非所问", "Especially with midterms coming up.",
     "So you are feeling more than just a little over that?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="data/annomi/mm_sft_final/holdout.jsonl")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--thresh", type=float, default=0.45)
    args = ap.parse_args()

    print("=== 6 条回归 ===")
    for note, ctx, reply in CASES:
        pen, ung = score_faithfulness(reply, ctx, args.thresh)
        print(f"  {note:<16} penalty={pen:.2f}  未grounded={ung}")

    import json
    rows = [json.loads(l) for l in open(args.holdout, encoding="utf-8")][:args.n]
    pens, flagged = [], 0
    for r in rows:
        gold = r["messages"][-1]["content"]
        us = [m["content"] for m in r["messages"][:-1] if m["role"] == "user"]
        cur = us[-1].split("]" + chr(10), 1)[-1] if us else ""
        hist = " ".join(m["content"] for m in r["messages"][:-1])
        pen, _ = score_faithfulness(gold, cur + " " + hist, args.thresh)
        pens.append(pen)
        flagged += pen > 0
    pens = np.array(pens)
    print(f"\n=== {len(rows)} 条 gold 误伤率(真人回复,penalty 应低)===")
    print(f"  平均 penalty = {pens.mean():.3f}   有任何flag的比例 = {flagged/len(rows)*100:.0f}%")
    print(f"  penalty 分布: =0 占 {(pens==0).mean()*100:.0f}%  >0.5 占 {(pens>0.5).mean()*100:.0f}%")


if __name__ == "__main__":
    main()
