# -*- coding: utf-8 -*-
"""open/closed 与 simple/complex 分类器: LogReg vs MLP(更充分训练), 冻结 MiniLM 嵌入, 会话分组CV。
回答'充分训练的分类器能到多高'。CPU 运行(不抢采样GPU)。"""
import csv
import numpy as np
from collections import defaultdict, Counter

FULL = "/root/Yuan-chat/data/annomi/AnnoMI-full.csv"
with open(FULL, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
by = defaultdict(list)
for r in rows: by[(r["transcript_id"], r["utterance_id"])].append(r)
def maj(rs,c): return Counter(x[c] for x in rs).most_common(1)[0][0]

def collect(exists_col, subtype_col, labels):
    X_txt, y, g = [], [], []
    for k,rs in by.items():
        if rs[0]["interlocutor"]!="therapist": continue
        if maj(rs,exists_col)!="True": continue
        sub=maj(rs,subtype_col)
        if sub not in labels: continue
        X_txt.append(rs[0]["utterance_text"]); y.append(sub); g.append(rs[0]["transcript_id"])
    return X_txt, np.array(y), np.array(g)

def embed(texts):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    return np.asarray(m.encode(texts, batch_size=256, show_progress_bar=False), dtype=np.float32)

def run(name, exists_col, subtype_col, labels):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, GroupKFold
    Xt, y_str, g = collect(exists_col, subtype_col, labels)
    base = max(Counter(y_str).values())/len(y_str)
    y = np.array([list(labels).index(v) for v in y_str])   # 整数标签(修 MLP+字符串的坑)
    X = embed(Xt)
    cv = GroupKFold(n_splits=5)
    print(f"\n=== {name} ===  n={len(y)} 分布={dict(Counter(y))} 多数类基线={base:.3f}")
    for tag, clf in [
        ("LogReg(线性)", LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced")),
        ("MLP(256,充分)", MLPClassifier(hidden_layer_sizes=(256,), max_iter=800, early_stopping=True,
                                        alpha=1e-3, random_state=0)),
        ("MLP(256,128)", MLPClassifier(hidden_layer_sizes=(256,128), max_iter=800, early_stopping=True,
                                        alpha=1e-3, random_state=0)),
    ]:
        acc = cross_val_score(clf, X, y, groups=g, cv=cv, scoring="accuracy")
        f1  = cross_val_score(clf, X, y, groups=g, cv=cv, scoring="f1_macro")
        print(f"  {tag:14s} acc={acc.mean():.3f}±{acc.std():.3f}  macroF1={f1.mean():.3f}")

run("question open/closed", "question_exists", "question_subtype", ("open","closed"))
run("reflection simple/complex", "reflection_exists", "reflection_subtype", ("simple","complex"))
