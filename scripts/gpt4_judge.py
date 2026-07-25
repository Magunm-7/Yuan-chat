"""
GPT-4 as judge:成对盲评 base/SFT/DPO 的 MI 咨询回复,算 win rate。在**本地**跑(连 OpenAI)。
API key 从环境变量 OPENAI_API_KEY 读,不硬编码、不经手。

消除位置偏差:每对随机 A/B 顺序;可选双向(--two_way)各评一次取一致。

用法(本地):
  export OPENAI_API_KEY=sk-...          # Windows: setx OPENAI_API_KEY "sk-..."
  pip install openai
  python scripts/gpt4_judge.py --responses data/annomi/responses_for_judge.jsonl \
      --pairs sft:dpo base:sft --model gpt-4o
"""
from __future__ import annotations
import os, re, json, argparse, random, time

SYSTEM = (
    "You are an expert evaluator of Motivational Interviewing (MI) counseling. "
    "MI best practice: reflective listening and open questions are good; giving advice, "
    "warnings, judgments, or lecturing is bad; the counselor must NOT fabricate facts the "
    "client did not say; responses should be brief and empathic. "
    "You will see the conversation so far and two candidate counselor replies (A and B). "
    "Decide which reply is the better MI response. Reply with exactly one token: A, B, or tie."
)


def render(msgs):
    lines = []
    for m in msgs:
        role = {"system": "SYSTEM", "user": "CLIENT", "assistant": "COUNSELOR"}.get(m["role"], m["role"])
        if m["role"] == "system":
            continue
        lines.append(f"{role}: {m['content']}")
    return "\n".join(lines)


def judge_once(client, model, msgs, reply_a, reply_b):
    user = (f"Conversation so far:\n{render(msgs)}\n\n"
            f"--- Reply A ---\n{reply_a}\n\n--- Reply B ---\n{reply_b}\n\n"
            f"Which reply is the better MI counselor response? Answer A, B, or tie.")
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=model, temperature=0, max_tokens=4,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}])
            out = r.choices[0].message.content.strip().lower()
            if out.startswith("a"):
                return "A"
            if out.startswith("b"):
                return "B"
            return "tie"
        except Exception as e:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", default="data/annomi/responses_for_judge.jsonl")
    ap.add_argument("--pairs", nargs="+", default=["sft:dpo", "base:sft"],
                    help="要比较的成对,格式 X:Y(问 X vs Y 谁更好)")
    ap.add_argument("--model", default=os.environ.get("JUDGE_MODEL", "gpt-4o"))
    ap.add_argument("--two_way", action="store_true", help="双向各评一次(更抗位置偏差,2x 成本)")
    ap.add_argument("--out", default="data/annomi/judge_results.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI()          # 自动读 OPENAI_API_KEY
    random.seed(args.seed)
    rows = [json.loads(l) for l in open(args.responses, encoding="utf-8")]
    print(f"{len(rows)} 条回复,judge={args.model},pairs={args.pairs}\n")

    log = []
    for spec in args.pairs:
        x, y = spec.split(":")
        wins = {x: 0, y: 0, "tie": 0}
        for r in rows:
            rx, ry = r.get(x, ""), r.get(y, "")
            if not rx or not ry:
                continue
            # 随机 A/B 位置,消除位置偏差
            swap = random.random() < 0.5
            a, b = (ry, rx) if swap else (rx, ry)
            v = judge_once(client, args.model, r["prompt_messages"], a, b)
            if v == "tie":
                winner = "tie"
            else:
                a_is_x = not swap
                winner = x if (v == "A") == a_is_x else y
            if args.two_way:                       # 反向再评一次
                v2 = judge_once(client, args.model, r["prompt_messages"], b, a)
                w2 = "tie" if v2 == "tie" else (y if (v2 == "A") == a_is_x else x)
                winner = winner if winner == w2 else "tie"   # 不一致判平
            wins[winner] += 1
            log.append({"pair": spec, "winner": winner, "user_text": r.get("user_text", "")})
        n = sum(wins.values())
        wr = wins[x] / n * 100 if n else 0
        print(f"  {x} vs {y}:  {x} 胜 {wins[x]} ({wr:.0f}%)  |  {y} 胜 {wins[y]} "
              f"({wins[y]/n*100:.0f}%)  |  平 {wins['tie']}   (n={n})")

    with open(args.out, "w", encoding="utf-8") as f:
        for e in log:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"\n逐条判决 -> {args.out}")
    print("判读:DPO vs SFT 胜率 >50% = DPO 净提升;<50% = DPO 净退步(反映↑但捏造/变长盖过)。")


if __name__ == "__main__":
    main()
