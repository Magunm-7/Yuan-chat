"""
DPO 训练(手写损失,不依赖 TRL)。

参考模型 = SFT 后的模型,不是原始 base —— 我们要约束的是"别偏离已学到的 MI 风格",
实现上把 SFT LoRA 先 merge 进权重,再挂一个新的 DPO LoRA;
关掉新 adapter 时得到的就是参考模型,**同一份权重两用,显存不翻倍**。

  loss = -log sigmoid( beta * [ (logp_pol(chosen)  - logp_ref(chosen))
                              - (logp_pol(reject)  - logp_ref(reject)) ] )

  python scripts/train_dpo.py --pairs data/annomi/dpo_pairs_full.jsonl \
      --sft_lora outputs/mm_sft/qwen7b_v3_2048 --out outputs/dpo/qwen7b_dpo
"""
from __future__ import annotations
import os, json, math, argparse, random
import torch
import torch.nn.functional as F


def build_example(tok, messages, reply, max_len):
    """返回 (input_ids, labels):labels 只在 reply 段计损失,prompt 段全 -100。"""
    try:
        p_txt = tok.apply_chat_template(messages, tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        f_txt = tok.apply_chat_template(messages + [{"role": "assistant", "content": reply}],
                                        tokenize=False, enable_thinking=False)
    except TypeError:
        p_txt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        f_txt = tok.apply_chat_template(messages + [{"role": "assistant", "content": reply}],
                                        tokenize=False)
    p_ids = tok(p_txt, return_tensors="pt", truncation=False)["input_ids"][0]
    f_ids = tok(f_txt, return_tensors="pt", truncation=False)["input_ids"][0]
    n_tgt = max(1, int(f_ids.shape[0]) - int(p_ids.shape[0]))
    ids = f_ids[-max_len:]                       # 左截断:丢最老历史,保住 reply
    labels = ids.clone()
    n_mask = int(ids.shape[0]) - n_tgt
    if n_mask > 0:
        labels[:n_mask] = -100
    return ids, labels


def seq_logprob(model, ids, labels, use_fast_logits=True):
    """整段 reply 的对数概率之和。只有尾部 reply 被监督,故只取尾部 logits(省显存)。"""
    ids = ids.unsqueeze(0)
    labels = labels.unsqueeze(0)
    n_tgt = int((labels != -100).sum())
    out = None
    if use_fast_logits and n_tgt > 0:
        for kw in ("logits_to_keep", "num_logits_to_keep"):
            try:
                out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                            return_dict=True, **{kw: n_tgt + 1})
                break
            except TypeError:
                out = None
        if out is not None and int(out.logits.shape[1]) != n_tgt + 1:
            out = None
    if out is None:
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True)
        shift_logits, shift_labels = out.logits[:, :-1, :], labels[:, 1:]
    else:
        shift_logits, shift_labels = out.logits[:, :-1, :], labels[:, -n_tgt:]
    logp = torch.log_softmax(shift_logits.float(), dim=-1)
    mask = shift_labels != -100
    tgt = shift_labels.clamp_min(0).unsqueeze(-1)
    tok_lp = torch.gather(logp, 2, tgt).squeeze(-1) * mask
    return tok_lp.sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/annomi/dpo_pairs_full.jsonl")
    ap.add_argument("--base", default="/root/autodl-tmp/models/Qwen2.5-7B-Instruct")
    ap.add_argument("--sft_lora", default="outputs/mm_sft/qwen7b_v3_2048")
    ap.add_argument("--out", default="outputs/dpo/qwen7b_dpo")
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(args.base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    print("[load] base + SFT LoRA -> merge(得到参考模型的权重)")
    lm = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    sft_dir = os.path.join(args.sft_lora, "lora_adapter")
    lm = PeftModel.from_pretrained(lm, sft_dir if os.path.isdir(sft_dir) else args.sft_lora)
    lm = lm.merge_and_unload()

    print("[init] 挂新的 DPO LoRA(B=0,初始输出与 SFT 完全一致)")
    lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(lm, LoraConfig(
        r=args.lora_r, lora_alpha=2 * args.lora_r, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.enable_input_require_grads()
    model.train()

    rows = [json.loads(l) for l in open(args.pairs, encoding="utf-8")]
    print(f"[data] {len(rows)} 偏好对")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    step, acc_n, acc_loss, acc_margin = 0, 0, 0.0, 0.0
    for ep in range(args.epochs):
        random.shuffle(rows)
        for r in rows:
            msgs = r["prompt_messages"]
            try:
                c_ids, c_lab = build_example(tok, msgs, r["chosen"], args.max_len)
                j_ids, j_lab = build_example(tok, msgs, r["rejected"], args.max_len)
            except Exception:
                continue
            c_ids, c_lab = c_ids.to(dev), c_lab.to(dev)
            j_ids, j_lab = j_ids.to(dev), j_lab.to(dev)

            with torch.autocast("cuda", dtype=torch.bfloat16):
                pol_c = seq_logprob(model, c_ids, c_lab)
                pol_j = seq_logprob(model, j_ids, j_lab)
                with torch.no_grad(), model.disable_adapter():      # 关掉 adapter = 参考模型
                    ref_c = seq_logprob(model, c_ids, c_lab)
                    ref_j = seq_logprob(model, j_ids, j_lab)
            margin = (pol_c - ref_c) - (pol_j - ref_j)
            loss = -F.logsigmoid(args.beta * margin)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            if torch.isfinite(gn):
                opt.step()
            step += 1; acc_n += 1
            acc_loss += float(loss); acc_margin += float(margin)
            if step % 50 == 0:
                print(f"  ep{ep+1} step {step}  loss={acc_loss/acc_n:.4f}  "
                      f"margin={acc_margin/acc_n:+.2f}  (margin>0 表示已偏向 chosen)", flush=True)
                acc_n, acc_loss, acc_margin = 0, 0.0, 0.0

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    print(f"saved -> {args.out}")
    print("注意:该 adapter 依赖 base+SFT 已 merge 的权重,推理需先 merge SFT LoRA 再加载它。")


if __name__ == "__main__":
    main()
