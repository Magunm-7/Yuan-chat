"""
验证 bart-large-mnli 能否把"捏造"和"忠实反映"分开 —— 整合进 reward 前的判别力 sanity check。
样例取自今晚 DPO 的真实输出。对比整句 NLI vs 子句级 min-entailment(后者应能抓半真半假的局部捏造)。

  premise = 来访者上下文, hypothesis = 咨询师回复(或其子句)
  entailment 高 = 上下文支持这句话; 低 = 无根据(捏造)

  python scripts/nli_faithfulness_test.py
"""
from __future__ import annotations
import re
import torch

MODEL = "facebook/bart-large-mnli"   # config: 0=contradiction 1=neutral 2=entailment

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
    parts = re.split(r"[.?!;]|\band\b|\bbut\b|\bso\b", text, flags=re.I)
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def main():
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(dev).eval()

    @torch.no_grad()
    def entail(premise, hypothesis):
        x = tok(premise, hypothesis, return_tensors="pt", truncation=True, max_length=256).to(dev)
        p = torch.softmax(model(**x).logits[0], dim=-1)
        return float(p[2])          # entailment 概率

    print(f"{'样例':<28} {'整句entail':>10} {'子句min':>9}  最不忠实子句")
    print("-" * 92)
    for note, ctx, reply in CASES:
        whole = entail(ctx, reply)
        clauses = split_clauses(reply) or [reply]
        ce = [(entail(ctx, c), c) for c in clauses]
        cmin, cworst = min(ce, key=lambda t: t[0])
        print(f"{note:<28} {whole:>10.3f} {cmin:>9.3f}  \"{cworst[:55]}\"")


if __name__ == "__main__":
    main()
