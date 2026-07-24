"""
实体级 grounding 烟雾测试。
核心:只抽"具体实体"(名词/数字/专有名词),逐个查它在上下文的语义场里能否被支撑。
放过抽象情感/态度词(feeling/concern/okay) -> 合理推断不受罚;抓凭空的具体事实(bus/ankle/three days)。

判据:捏造条应列出未 grounded 的具体实体;忠实条应几乎全 grounded;
      合理推断("drunk")不该被误判为捏造。

  python scripts/entity_grounding_test.py --thresh 0.45
"""
from __future__ import annotations
import argparse
import numpy as np

CASES = [
    ("局部捏造(跳公交崴脚)",
     "I was at a pub crawl last night, you have a few drinks and I thought it was a good idea.",
     "So you were doing a pub crawl and you jumped off the stairs of the bus and you landed on your ankle?"),
    ("捏造数字(三天一周)",
     "me and my friends party on the weekend, we are all university students, just have a few drinks.",
     "So you are having a few drinks on the weekends, it sounds like you are drinking about three days a week."),
    ("半真半假(七八杯真/从学校回家喝假)",
     "I can drink probably like seven or eight drinks if I am going hard.",
     "So you are drinking seven to eight drinks on a night out and a couple more when you get home from school."),
    ("忠实反映+合理推进(情感延伸)",
     "Well, it is normal.",
     "It is normal? So it is okay to be a little bit drunk?"),
    ("忠实复述",
     "me and my friends party on the weekend, just have a few drinks.",
     "So a few drinks is nothing new to you?"),
    ("答非所问(与 midterms 无关)",
     "Especially with midterms coming up.",
     "So you are feeling more than just a little over that? You are feeling a lot more?"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresh", type=float, default=0.45, help="max grounding 低于此=未grounded")
    args = ap.parse_args()

    import spacy
    from sentence_transformers import SentenceTransformer
    nlp = spacy.load("en_core_web_sm")
    st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def norm(xs):
        v = np.asarray(st.encode(xs), dtype=np.float32)
        return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)

    def concrete_entities(text):
        """只留具体实体:名词/专有名词/数字,去停用词。抽象情感态度词大多是形容词/被停用词表覆盖。"""
        doc = nlp(text)
        out = []
        for t in doc:
            if t.pos_ in ("NOUN", "PROPN", "NUM") and not t.is_stop and len(t.lemma_) > 1:
                out.append(t.lemma_.lower())
        return list(dict.fromkeys(out))

    for note, ctx, reply in CASES:
        r_ents = concrete_entities(reply)
        c_ents = concrete_entities(ctx)
        print(f"\n=== {note}")
        print(f"  回复: {reply}")
        print(f"  上下文实体: {c_ents}")
        if not r_ents:
            print("  回复无具体实体")
            continue
        if not c_ents:
            grounds = [(e, 0.0) for e in r_ents]
        else:
            sims = norm(r_ents) @ norm(c_ents).T
            grounds = [(e, float(sims[i].max())) for i, e in enumerate(r_ents)]
        ung = [e for e, s in grounds if s < args.thresh]
        print("  实体 grounding: " + "  ".join(f"{e}={s:.2f}" for e, s in grounds))
        print(f"  >>> 未 grounded(<{args.thresh}) = {ung}  →  {'含捏造' if ung else '忠实'}")


if __name__ == "__main__":
    main()
