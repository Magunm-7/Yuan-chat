# -*- coding: utf-8 -*-
"""训练并保存 open/closed 分类器(LogReg 冻结MiniLM, 全量拟合用于部署)。CPU。"""
import csv, numpy as np
from collections import defaultdict, Counter

FULL = "/root/Yuan-chat/data/annomi/AnnoMI-full.csv"
OUT = "/root/Yuan-chat/outputs/evaluator/oc_clf.npz"

with open(FULL, encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
by = defaultdict(list)
for r in rows: by[(r["transcript_id"], r["utterance_id"])].append(r)
def maj(rs,c): return Counter(x[c] for x in rs).most_common(1)[0][0]

Xt, y = [], []
for k,rs in by.items():
    if rs[0]["interlocutor"]!="therapist": continue
    if maj(rs,"question_exists")!="True": continue
    sub=maj(rs,"question_subtype")
    if sub not in ("open","closed"): continue
    Xt.append(rs[0]["utterance_text"]); y.append(sub)
print(f"训练样本 {len(y)}  {dict(Counter(y))}")

from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
X = np.asarray(m.encode(Xt, batch_size=256, show_progress_bar=False), dtype=np.float32)
KEYS=["open","closed"]
yi = np.array([KEYS.index(v) for v in y])
clf = LogisticRegression(max_iter=3000, C=2.0, class_weight="balanced").fit(X, yi)
import os; os.makedirs(os.path.dirname(OUT), exist_ok=True)
np.savez(OUT, coef=clf.coef_, intercept=clf.intercept_, keys=np.array(KEYS))
print("saved ->", OUT, "| 训练集自评 acc", (clf.predict(X)==yi).mean())
