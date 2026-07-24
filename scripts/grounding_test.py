"""
NLI 失败后的替代:子句级语义 grounding。
NLI 判"逻辑蕴含"与 MI 反映(转述/疑问/推断)不对齐;改用 MiniLM 语义相似度 ——
只要求"子句是否与上下文语义相关",捏造子句(引入上下文没有的具体实体)相似度低。
取回复各子句与上下文的 min cosine 作为忠实性代理。复用现成 MiniLM,不加新模型。

  python scripts/grounding_test.py
"""
from __future__ import annotations
import re
import numpy as np

CASES = [
    ("局部捏造(跳公交崴脚)",
     "I was at a pub crawl last night, you have a few drinks and I thought it was a good idea.",
     "So you were doing a pub crawl and you jumped off the stairs of the bus and you landed on your ankle?"),
    ("捏造数字(三天一周)",
     "me and my friends party on the weekend, we are all university students, just have a few drinks.",
     "So you are having a few drinks on the weekends, so it sounds like you are drinking about three days a week."),
    ("半真半假(七八杯真/从学校回家喝假)",
     "I can drink probably like seven or eight drinks if I am going hard.",
     "So you are drinking seven to eight drinks on a night out and then a couple more when you get home from school."),
    ("忠实反映+合理推进",
     "Well, it is normal.",
     "It is normal? So it is okay to be a little bit drunk?"),
    ("忠实复述",
     "me and my friends party on the weekend, just have a few drinks.",
     "So a few drinks is nothing new to you?"),
    ("答非所问(与 midterms 无关)",
     "Especially with midterms coming up.",
     "So you are feeling more than just a little over that? You are feeling a lot more?"),
]


def split_clauses(text):
    parts = re.split(r"[.?!;,]|\band\b|\bbut\b|\bso\b", text, flags=re.I)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def main():
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def emb(xs):
        v = np.asarray(st.encode(xs), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    print(f"{'样例':<28} {'整句rel':>8} {'子句min':>8}  最不相关子句")
    print("-" * 96)
    for note, ctx, reply in CASES:
        cv = emb([ctx])[0]
        whole = float(emb([reply])[0] @ cv)
        clauses = split_clauses(reply) or [reply]
        sims = emb(clauses) @ cv
        j = int(sims.argmin())
        print(f"{note:<28} {whole:>8.3f} {float(sims[j]):>8.3f}  \"{clauses[j][:52]}\"")


if __name__ == "__main__":
    main()
