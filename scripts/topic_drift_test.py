"""
对照实验:system prompt 的话题约束能否抑制"幻想话题"。

背景:训练样本只有 4.2% 是会话开场,模型在冷启动(用户没给话题)时会从训练分布里
抽高频话题填充(酒精 8639 / 吸烟 4212 / 体重 3331 次)。这里量化两个指标:
  - 话题漂移率:回复提到成瘾话题、而用户输入从未提及
  - 行为分布:reflection / question / therapist_input / other(用行为分类器判定)

通过 HTTP 调用已在跑的 demo 服务,复用其已加载模型,不额外占显存。
  python scripts/topic_drift_test.py --n 3
"""
from __future__ import annotations
import re, json, argparse, urllib.request
import numpy as np

API = "http://localhost:8000/infer"

BASE_PROMPT = ("You are an experienced motivational interviewing (MI) counselor in an ongoing "
               "session. Given the conversation so far and the client's latest utterance (with "
               "the nonverbal cues provided), reply with a single brief, empathic counselor "
               "response that fits the topic and gently supports the client's own motivation to change.")

CONSTRAINED_PROMPT = BASE_PROMPT + (
    "\n\nStrict rules:\n"
    "1. Respond ONLY to what the client actually said. NEVER introduce a topic "
    "(alcohol, smoking, weight, drugs, exercise) that the client has not mentioned.\n"
    "2. If the client has not yet raised a concern, ask one open question inviting them "
    "to share what brought them here.\n"
    "3. Never give advice, opinions, warnings, or judgements. Only reflect and ask.\n"
    "4. One or two short sentences."
)

# 冷启动:完全没有话题锚点(最容易诱发幻想)
COLD = ["Hello.", "Hi there.", "I'm feeling okay today.", "Not much, just tired.",
        "I don't know where to start.", "It's been a rough week.",
        "I guess I'm fine.", "Nice to meet you."]
# 对照:输入自带明确话题
WARM = ["I've been drinking almost every night lately.",
        "I tried to quit smoking last month but I failed again."]

TOPIC_RE = re.compile(r"\b(alcohol|drink|drinking|drank|beer|wine|booze|"
                      r"smok\w*|cigarette|vape|nicotine|"
                      r"weight|diet|obes\w*|calorie|"
                      r"drug|cocaine|heroin|cannabis|opioid)\b", re.I)


def ask(text, sys_prompt, temperature, sid):
    body = json.dumps({"session_id": sid, "text": text, "temperature": temperature,
                       "max_new_tokens": 96, "system_prompt": sys_prompt}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def run(tag, sys_prompt, inputs, args, behav):
    drift, replies = 0, []
    for i, t in enumerate(inputs):
        for k in range(args.n):
            d = ask(t, sys_prompt, args.temperature, f"drift-{tag}-{i}-{k}-{np.random.randint(1e6)}")
            rep = d.get("reply", "")
            replies.append(rep)
            said = bool(TOPIC_RE.search(t))          # 用户是否提到过话题
            used = bool(TOPIC_RE.search(rep))        # 回复是否用了话题词
            if used and not said:
                drift += 1
    labs, _ = behav(replies)
    n = len(replies)
    dist = {k: sum(1 for l in labs if l == k) / n * 100
            for k in ["reflection", "question", "therapist_input", "other"]}
    wl = np.mean([len(r.split()) for r in replies])
    print(f"  {tag:22s} 漂移={drift}/{n} ({drift/n*100:.0f}%)  " +
          "  ".join(f"{k[:6]}={v:4.1f}%" for k, v in dist.items()) + f"  词数={wl:.1f}")
    return replies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3, help="每个输入采样几次")
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from behaviour_scorer import predict
    behav = lambda texts: predict(texts, "cpu")

    print("=== 冷启动输入(无话题锚点)===")
    a = run("原 prompt", BASE_PROMPT, COLD, args, behav)
    b = run("加话题约束", CONSTRAINED_PROMPT, COLD, args, behav)

    print("\n=== 对照:输入自带话题 ===")
    run("原 prompt", BASE_PROMPT, WARM, args, behav)
    run("加话题约束", CONSTRAINED_PROMPT, WARM, args, behav)

    print("\n=== 冷启动样例对比 ===")
    for i, t in enumerate(COLD[:4]):
        print(f"\n-- 输入: {t}")
        print(f"   [原  ] {a[i*args.n][:130]}")
        print(f"   [约束] {b[i*args.n][:130]}")


if __name__ == "__main__":
    main()
