# -*- coding: utf-8 -*-
"""SMOKE TEST 2 (服务器, 基于现有产物): full 子类能否从文本可靠分类?
可行就打到 base/sft/dpo 回复上, 验 'base 赢在 complex反映 + open提问'。
不重训任何大模型, 只训两个 LogReg 探针 (秒级)。"""
import csv, json, sys
import numpy as np
from collections import Counter, defaultdict

FULL = "/root/Yuan-chat/data/annomi/AnnoMI-full.csv"
RESP = "/root/Yuan-chat/data/annomi/responses_14b.jsonl"

def load_full():
    with open(FULL, encoding="utf-8") as f:
        return list(csv.DictReader(f))

def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return np.asarray(m.encode(texts, batch_size=128, show_progress_bar=False), dtype=np.float32)

def dedup_majority(full):
    """每个唯一(transcript,utterance)取多数标签, 返回 therapist 句 list。"""
    by = defaultdict(list)
    for r in full:
        by[(r["transcript_id"], r["utterance_id"])].append(r)
    out = []
    for k, rows in by.items():
        if rows[0]["interlocutor"] != "therapist":
            continue
        def maj(c): return Counter(r[c] for r in rows).most_common(1)[0][0]
        out.append({
            "sid": rows[0]["transcript_id"],
            "text": rows[0]["utterance_text"],
            "refl": maj("reflection_exists"), "refl_sub": maj("reflection_subtype"),
            "q": maj("question_exists"), "q_sub": maj("question_subtype"),
        })
    return out

def cv_probe(texts, y, groups, name):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold
    X = embed(texts)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    cv = GroupKFold(n_splits=5)
    acc = cross_val_score(clf, X, np.array(y), groups=np.array(groups), cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, np.array(y), groups=np.array(groups), cv=cv, scoring="f1_macro")
    # 基线: 多数类
    base = max(Counter(y).values()) / len(y)
    print(f"  [{name}] n={len(y)} 分布={dict(Counter(y))}")
    print(f"     CV accuracy={acc.mean():.3f}±{acc.std():.3f}  macroF1={f1.mean():.3f}±{f1.std():.3f}  (多数类基线={base:.3f})")
    clf.fit(X, np.array(y))
    return clf

def main():
    full = load_full()
    ther = dedup_majority(full)
    print("="*66)
    print("ST2-A  子类可分性 (生死闸: 分不出就当场死)")
    print("="*66)

    # reflection: simple vs complex
    refl = [t for t in ther if t["refl"]=="True" and t["refl_sub"] in ("simple","complex")]
    clf_refl = cv_probe([t["text"] for t in refl],
                        [t["refl_sub"] for t in refl],
                        [t["sid"] for t in refl], "reflection simple/complex")
    # question: open vs closed
    q = [t for t in ther if t["q"]=="True" and t["q_sub"] in ("open","closed")]
    clf_q = cv_probe([t["text"] for t in q],
                     [t["q_sub"] for t in q],
                     [t["sid"] for t in q], "question open/closed")

    print()
    print("="*66)
    print("ST2-B  打到 base/sft/dpo 回复上 (60条holdout, 指示性)")
    print("="*66)
    resp = [json.loads(l) for l in open(RESP, encoding="utf-8")]
    # 先用现有4类 behaviour 分类器判是不是 reflection/question, 再用子类探针细分
    # 简化: 直接对每条回复问两个探针 —— 但只在"像反映/像提问"时才有意义。
    # 这里直接给全部回复打两个子类概率, 报聚合(指示性, 非严格pipeline)。
    for who in ["gold","base","sft","dpo"]:
        texts = [r[who] for r in resp]
        Xr = embed(texts)
        # reflection complex 概率
        pr = clf_refl.predict_proba(Xr)
        ci = list(clf_refl.classes_).index("complex")
        complex_rate = (pr[:,ci] > 0.5).mean()
        # question open 概率
        pq = clf_q.predict_proba(Xr)
        oi = list(clf_q.classes_).index("open")
        open_rate = (pq[:,oi] > 0.5).mean()
        wl = np.mean([len(t.split()) for t in texts])
        print(f"  {who:5s} 词长{wl:5.1f}  探针判 complex反映 {100*complex_rate:4.0f}%   探针判 open提问 {100*open_rate:4.0f}%")
    print("\n  注: ST2-B 是把子类探针无差别打到所有回复上, 指示性。若 base 的 complex/open")
    print("  明显高于 dpo, 即支持 'base 赢在MI技术含量' 的故事; 严格版需先过 4类门再细分。")

if __name__ == "__main__":
    main()
