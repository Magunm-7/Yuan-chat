"""
MI 咨询师行为分类器(reflection / question / therapist_input / other)。

用途有二:
  1) 诊断:量化当前微调模型的生成偏向哪种行为(是否在说教)
  2) 若走 DPO:作为 turn 级 reward —— 高质量会话 reflection 占 29.1% vs 低质量 6.3%,
     therapist_input 则是 8.0% vs 35.4%,信号强且是现成标注。

  python scripts/behaviour_scorer.py --train        # 训练 + 交叉验证
  python scripts/behaviour_scorer.py --diagnose     # 对比 gold 与模型生成的行为分布
"""
from __future__ import annotations
import os, json, argparse
import numpy as np

CKPT = "outputs/evaluator/behaviour_clf.npz"
KEYS = ["reflection", "question", "therapist_input", "other"]
# MI 原则:多反映、多开放式提问、少灌输 -> 用于 DPO 时的标量奖励
REWARD = {"reflection": 1.0, "question": 0.6, "other": 0.2, "therapist_input": -0.5}


def _embed(texts, device="cpu"):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    return np.asarray(m.encode(texts, batch_size=128, show_progress_bar=False), dtype=np.float32)


def train(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score, GroupKFold

    rows = [json.loads(l) for l in open("data/annomi/turns_labeled.jsonl", encoding="utf-8")]
    rows = [r for r in rows if (r.get("reply_behaviour") in KEYS) and (r.get("therapist_reply") or "").strip()]
    X_txt = [r["therapist_reply"] for r in rows]
    y = np.array([KEYS.index(r["reply_behaviour"]) for r in rows])
    groups = np.array([r["session_id"] for r in rows])          # 按会话分组, 防泄漏
    print(f"样本 {len(rows)}  类别分布: " +
          "  ".join(f"{k}={int((y==i).sum())}" for i, k in enumerate(KEYS)))

    X = _embed(X_txt, args.device)
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced")
    cv = GroupKFold(n_splits=5)
    acc = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="accuracy")
    f1 = cross_val_score(clf, X, y, groups=groups, cv=cv, scoring="f1_macro")
    print(f"会话分组 5 折 CV:  accuracy={acc.mean():.3f}±{acc.std():.3f}   macro-F1={f1.mean():.3f}±{f1.std():.3f}")

    clf.fit(X, y)
    os.makedirs(os.path.dirname(CKPT), exist_ok=True)
    np.savez(CKPT, coef=clf.coef_, intercept=clf.intercept_, keys=np.array(KEYS))
    print("saved ->", CKPT)


def _load_clf():
    d = np.load(CKPT, allow_pickle=True)
    return d["coef"], d["intercept"], [str(k) for k in d["keys"]]


def predict(texts, device="cpu"):
    coef, intercept, keys = _load_clf()
    X = _embed(texts, device)
    logits = X @ coef.T + intercept
    e = np.exp(logits - logits.max(1, keepdims=True))
    p = e / e.sum(1, keepdims=True)
    idx = p.argmax(1)
    return [keys[i] for i in idx], p


def diagnose(args):
    """对比三组回复的行为分布:gold 真人 / 当前微调模型 / base 未微调。"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    rows = [json.loads(l) for l in open("data/annomi/mm_sft_final/holdout.jsonl", encoding="utf-8")]
    rows = rows[:args.n]
    golds = [r["messages"][-1]["content"] for r in rows]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev).eval()

    def gen_all(model, tag):
        outs = []
        for i, r in enumerate(rows):
            msgs = r["messages"][:-1]
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                            enable_thinking=False)
            except TypeError:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            ids = tok(t, return_tensors="pt", truncation=True, max_length=1536).input_ids.to(dev)
            with torch.no_grad():
                o = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=True,
                                   temperature=args.temperature, top_p=0.85,
                                   repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            outs.append(tok.decode(o[0, ids.shape[1]:], skip_special_tokens=True).strip())
            if (i + 1) % 10 == 0:
                print(f"  [{tag}] {i+1}/{len(rows)}", flush=True)
        return outs

    base_out = gen_all(lm, "base")
    ft = PeftModel.from_pretrained(lm, os.path.join(args.lora, "lora_adapter")
                                   if os.path.isdir(os.path.join(args.lora, "lora_adapter"))
                                   else args.lora).eval()
    ft_out = gen_all(ft, "finetuned")

    print("\n=== 行为分布对比(分类器预测)===")
    for name, texts in (("gold 真人咨询师", golds), ("微调模型", ft_out), ("base 未微调", base_out)):
        labs, _ = predict(texts, "cpu")
        n = len(labs)
        dist = {k: sum(1 for l in labs if l == k) / n * 100 for k in KEYS}
        rew = np.mean([REWARD[l] for l in labs])
        wl = np.mean([len(t.split()) for t in texts])
        print(f"  {name:16s} " + "  ".join(f"{k[:6]}={dist[k]:5.1f}%" for k in KEYS) +
              f"   | 平均奖励={rew:+.3f}  平均词数={wl:.1f}")

    print("\n=== 样例(前 3 条)===")
    for i in range(min(3, len(rows))):
        print(f"\n-- 输入: {rows[i]['messages'][-2]['content'][:90]}")
        for name, t in (("gold", golds[i]), ("微调", ft_out[i]), ("base", base_out[i])):
            lab, _ = predict([t], "cpu")
            print(f"   [{name:4s}|{lab[0]:15s}] {t[:150]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    if args.train:
        train(args)
    if args.diagnose:
        diagnose(args)


if __name__ == "__main__":
    main()
