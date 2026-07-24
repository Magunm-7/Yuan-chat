"""
State-controllability check for the state-tag generator (pure-text LoRA).

For each holdout turn we hold the input fixed and flip ONE dimension of the
in-prompt state tag (e.g. change-readiness: low <-> high), same seed, and compare
the two generated replies. Because the tag is text (hard), the model must read it,
so this is controllable by construction; this just shows it on real turns.

  python scripts/state_ctrl.py --base <B> --lora outputs/mm_sft/qwen7b_tag \
      --index data/annomi/mm_sft_tag/holdout.jsonl --dim change-readiness --vals low,high
"""
from __future__ import annotations
import json, argparse, re, copy
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def gen(tok, model, msgs, max_new, seed, dev):
    # enable_thinking=False keeps the generation prompt identical to training
    # (Qwen3 puts an empty <think> block there; Qwen2.5 ignores the kwarg).
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids.to(dev)
    torch.manual_seed(seed)
    out = model.generate(ids, max_new_tokens=max_new, do_sample=True, temperature=0.7,
                         top_p=0.9, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def set_level(content, dim, level):
    return re.sub(dim + r": (low|moderate|high|negative|neutral|positive)",
                  f"{dim}: {level}", content)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--lora", required=True)
    ap.add_argument("--index", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--min_gold_words", type=int, default=8)
    ap.add_argument("--dim", default="change-readiness")
    ap.add_argument("--vals", default="low,high")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    from peft import PeftModel
    lm = PeftModel.from_pretrained(lm, args.lora).to(dev).eval()

    items = [json.loads(l) for l in open(args.index, encoding="utf-8")]
    items = [it for it in items
             if len(it["messages"][-1]["content"].split()) >= args.min_gold_words]
    vals = args.vals.split(",")
    diff = 0
    shown = 0
    for it in items[:args.n]:
        msgs = it["messages"]
        gold = msgs[-1]["content"] if msgs[-1]["role"] == "assistant" else ""
        base_msgs = msgs[:-1]  # strip target
        # current user = last user message (tag + client)
        ui = max(j for j, m in enumerate(base_msgs) if m["role"] == "user")
        cur = base_msgs[ui]["content"]
        client_line = cur.split("]\n", 1)[-1] if "]" in cur else cur
        shown += 1
        print(f"\n=== sample {shown} ===")
        print("CLIENT:", client_line[:120])
        print("GOLD  :", gold[:120])
        outs = []
        for lv in vals:
            m2 = copy.deepcopy(base_msgs)
            m2[ui]["content"] = set_level(cur, args.dim, lv)
            g = gen(tok, lm, m2, args.max_new_tokens, args.seed + shown, dev)
            outs.append(g)
            print(f"  [{args.dim}={lv}] {g}")
        if len(set(outs)) > 1:
            diff += 1
    print(f"\n=== controllability: {diff}/{shown} turns changed when flipping '{args.dim}' ===")


if __name__ == "__main__":
    main()
