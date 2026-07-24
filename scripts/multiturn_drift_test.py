"""
多轮话题漂移测试:验证"雪球效应"——单轮漂移概率只有 8%,但一旦踩中,
模型会把自己编造的话题当既定事实继续推进,污染整段对话。

指标:
  首漂轮次   第几轮首次冒出用户没提过的成瘾话题
  粘性       首漂之后的轮次里, 仍继续讲该话题的比例(能否自我纠正)
  整段污染率 至少漂移一次的对话占比

  python scripts/multiturn_drift_test.py --dialogues 6 --turns 8
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

# 低信息量寒暄序列:用户始终不提任何具体问题(复现截图里的场景)
USER_TURNS = ["Hello.", "I'm feeling good.", "Not much really.", "Yeah.",
              "I don't know.", "Maybe.", "I guess so.", "Okay."]

TOPIC_RE = re.compile(r"\b(alcohol|drink|drinking|drank|beer|wine|booze|"
                      r"smok\w*|cigarette|vape|nicotine|"
                      r"weight|diet|obes\w*|calorie|"
                      r"drug|cocaine|heroin|cannabis|opioid)\b", re.I)


def ask(text, sys_prompt, temperature, sid):
    body = json.dumps({"session_id": sid, "text": text, "temperature": temperature,
                       "max_new_tokens": 96, "system_prompt": sys_prompt}).encode()
    req = urllib.request.Request(API, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())


def one_dialogue(sys_prompt, args, sid):
    """跑一整段对话, 返回每轮 (用户输入, 回复, 是否含话题词)。"""
    out = []
    for t in USER_TURNS[:args.turns]:
        d = ask(t, sys_prompt, args.temperature, sid)
        rep = d.get("reply", "") or d.get("error", "")
        out.append((t, rep, bool(TOPIC_RE.search(rep))))
    return out


def summarize(tag, dialogues):
    first, sticky, polluted = [], [], 0
    for turns in dialogues:
        flags = [d for _, _, d in turns]
        if any(flags):
            polluted += 1
            i = flags.index(True)
            first.append(i + 1)
            after = flags[i + 1:]
            if after:
                sticky.append(sum(after) / len(after))
    n = len(dialogues)
    fm = f"{np.mean(first):.1f}" if first else "—"
    sm = f"{np.mean(sticky)*100:.0f}%" if sticky else "—"
    print(f"  {tag:14s} 整段污染 {polluted}/{n} ({polluted/n*100:.0f}%)   "
          f"平均首漂轮次={fm}   首漂后粘性={sm}")
    return polluted / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dialogues", type=int, default=6)
    ap.add_argument("--turns", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    args = ap.parse_args()

    print(f"每组 {args.dialogues} 段对话 × {args.turns} 轮,用户全程只说无信息量的寒暄\n")
    results = {}
    for tag, sp in (("原 prompt", BASE_PROMPT), ("加话题约束", CONSTRAINED_PROMPT)):
        ds = [one_dialogue(sp, args, f"mt-{tag}-{i}-{np.random.randint(1e6)}")
              for i in range(args.dialogues)]
        results[tag] = ds
        summarize(tag, ds)

    print("\n=== 完整对话样例(原 prompt,第一段)===")
    for i, (u, r, d) in enumerate(results["原 prompt"][0], 1):
        mark = "  ← 漂移" if d else ""
        print(f"  {i}. 来访者: {u}")
        print(f"     咨询师: {r[:120]}{mark}")

    print("\n=== 完整对话样例(加约束,第一段)===")
    for i, (u, r, d) in enumerate(results["加话题约束"][0], 1):
        mark = "  ← 漂移" if d else ""
        print(f"  {i}. 来访者: {u}")
        print(f"     咨询师: {r[:120]}{mark}")


if __name__ == "__main__":
    main()
