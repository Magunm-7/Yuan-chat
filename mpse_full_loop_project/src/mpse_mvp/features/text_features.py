import re
import numpy as np

def text_quality(text: str):
    t = (text or "").strip()
    if not t:
        return 0.1
    # simple heuristic: longer text generally more informative
    L = len(t)
    return float(np.clip(L / 40.0, 0.2, 1.0))

def basic_text_feats(text: str):
    t = (text or "").strip()
    # simple bag-like features: length, punctuation, negations
    L = len(t)
    neg = len(re.findall(r"不|没|无|不能|难", t))
    ques = t.count("?") + t.count("？")
    ex = t.count("!") + t.count("！")
    return np.array([L/100.0, neg/10.0, ques/5.0, ex/5.0], dtype=np.float32)
