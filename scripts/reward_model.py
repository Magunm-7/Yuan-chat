"""
复合 reward(DPO / best-of-n 的打分器)。不训练 RM,而是组合三个可计算信号:

  behaviour    MI 行为类型(已训分类器): reflection +1.0 / question +0.6 / other +0.2
               / therapist_input -0.5   —— 管"说话方式对不对"
  relevance    回复与来访者最新一句的语义相似度(MiniLM cosine) —— 管"有没有答非所问"
  fabrication  出现 "you said / we talked about / last time" 等回指、但对话历史里
               并无对应内容 —— 管"有没有捏造来访者说过的话"

单看任何一项都能被钻空子:只看 behaviour,聊错话题的漂亮反映照样满分;
只看 relevance,鹦鹉学舌式复读得分最高。

  python scripts/reward_model.py --demo     # 采样候选并按 reward 排序, 人工核验
"""
from __future__ import annotations
import re, json, argparse, urllib.request
import numpy as np

API = "http://localhost:8000/infer"
W = {"behaviour": 1.0, "relevance": 0.4, "fabrication": 1.2}

# relevance 做成"门控"而非加项:答非所问时把行为得分打折,
# 否则一句漂亮但聊错话题的反映会靠 behaviour 单项硬撑到高分(实测 rel=0.01 仍排第三)。
REL_MID, REL_SLOPE = 0.28, 12.0

# 回指:声称来访者说过 / 双方谈过某事
ANAPHORA_RE = re.compile(
    r"\b(you (said|mentioned|told me|were saying)|we (talked|discussed|spoke)|"
    r"last time|earlier you|as you (said|mentioned))\b", re.I)

# 无根据地引入具体情境事实(实测捏造多半不走回指句式,而是直接陈述)
UNGROUNDED_RE = re.compile(
    r"\b(referral|referred (by|from)|another (counselor|therapist|doctor)|"
    r"your (doctor|physician|husband|wife|partner|boss|family)|"
    r"(the|your) (appointment|schedule|test results|chart|paperwork)|"
    r"i (noticed|can see|see) (that )?you('re| are)|"
    r"\d+\s*(days?|weeks?|months?|years?|drinks?|cigarettes?|pounds?|kilos?|kilograms?))\b", re.I)

_ST = {}


def _embed(texts):
    if "m" not in _ST:
        from sentence_transformers import SentenceTransformer
        _ST["m"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    v = np.asarray(_ST["m"].encode(texts), dtype=np.float32)
    return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-8)


def fabrication_penalty(reply: str, context: list[str]) -> float:
    """
    两类捏造:
      a) 回指来访者说过的话("you said…"),但上下文里查无此事
      b) 直接陈述一个上下文没提供的具体情境事实(转介、医生、天数…)
    有嫌疑时,用与上下文的语义相似度判断能否被支撑;上下文为空则必然是编的。
    """
    hit = bool(ANAPHORA_RE.search(reply)) or bool(UNGROUNDED_RE.search(reply))
    if not hit:
        return 0.0
    ctx = [c for c in (context or []) if c.strip()]
    if not ctx:
        return 1.0
    v = _embed([reply] + ctx)
    sim = float((v[1:] @ v[0]).max())
    return float(np.clip((0.5 - sim) / 0.5, 0.0, 1.0))


def score(user_text: str, replies: list[str], history: list[str] | None = None):
    from behaviour_scorer import predict, REWARD
    history = history or []
    labs, _ = predict(replies, "cpu")
    v = _embed([user_text] + replies)
    rel = (v[1:] @ v[0])                       # cosine(reply, user_text)
    out = []
    for r, lab, rl in zip(replies, labs, rel):
        fab = fabrication_penalty(r, [user_text] + history)
        gate = 1.0 / (1.0 + np.exp(-(float(rl) - REL_MID) * REL_SLOPE))   # 答非所问 -> 行为分打折
        total = (W["behaviour"] * REWARD[lab] * gate + W["relevance"] * float(rl)
                 - W["fabrication"] * fab)
        out.append({"reply": r, "behaviour": lab, "b": REWARD[lab], "gate": float(gate),
                    "rel": float(rl), "fab": fab, "total": float(total)})
    return out


def ask(text, temperature, sid, n_tok=96):
    body = json.dumps({"session_id": sid, "text": text, "temperature": temperature,
                       "max_new_tokens": n_tok}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read()).get("reply", "")


def demo(args):
    INPUTS = [
        "I had way too much to drink at the party and embarrassed myself.",
        "Hello.",
        "I don't really see the problem, everyone my age drinks like this.",
        "I tried to quit last month but I only lasted four days.",
    ]
    for t in INPUTS:
        cands = []
        for k in range(args.k):
            r = ask(t, args.temperature, f"rw-{abs(hash(t))%9999}-{k}-{np.random.randint(1e6)}")
            if r and r not in cands:
                cands.append(r)
        if not cands:
            continue
        rows = sorted(score(t, cands), key=lambda d: -d["total"])
        print(f"\n=== 来访者: {t}")
        for d in rows:
            print(f"  {d['total']:+.3f} = b({d['behaviour'][:6]:6s}){d['b']:+.1f}"
                  f"×gate{d['gate']:.2f} rel{d['rel']:+.2f} fab{-d['fab']:+.2f}"
                  f"  | {d['reply'][:105]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--k", type=int, default=8, help="每个输入采样几个候选")
    ap.add_argument("--temperature", type=float, default=0.9, help="高温以拉开候选差异")
    args = ap.parse_args()
    if args.demo:
        demo(args)


if __name__ == "__main__":
    main()
