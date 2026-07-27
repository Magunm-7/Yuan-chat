"""验证: NaN 是否由「16 个全零 prefix token」引起(Qwen3 的 QK-Norm 对全零输入更敏感)。
用裸 LM(带 LoRA)测, 排除 MultiModalPrefixLM 封装的干扰。"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm

BASE = "/root/autodl-tmp/models/Qwen3-14B"
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                   task_type="CAUSAL_LM",
                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
lm.train()

ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=512)
b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
ids = b["input_ids"].cuda(); attn = b["attention_mask"].cuda(); lbl = b["labels"].cuda()
H = lm.config.hidden_size
K = 16


def test(tag, prefix):
    lm.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        emb = lm.get_input_embeddings()(ids)
        if prefix is None:
            ie, am, lb = emb, attn, lbl
        else:
            p = prefix.to(emb.dtype)
            ie = torch.cat([p, emb], dim=1)
            am = torch.cat([torch.ones((1, K), dtype=attn.dtype, device=attn.device), attn], dim=1)
            lb = torch.cat([torch.full((1, K), -100, dtype=lbl.dtype, device=lbl.device), lbl], dim=1)
        loss = lm(inputs_embeds=ie, attention_mask=am, labels=lb).loss
    loss.backward()
    tot = bad = 0
    for n, p_ in lm.named_parameters():
        if p_.requires_grad and p_.grad is not None:
            tot += 1
            bad += int(bool(torch.isnan(p_.grad).any() or torch.isinf(p_.grad).any()))
    print(f"[{tag:22s}] loss={float(loss):.5f}  坏梯度={bad}/{tot}  -> "
          f"{'OK 梯度干净' if bad == 0 else 'NaN'}", flush=True)
    lm.zero_grad(set_to_none=True)


test("无 prefix", None)
test("全零 prefix x16", torch.zeros(1, K, H, device="cuda"))
test("随机 prefix x16", torch.randn(1, K, H, device="cuda") * 0.02)
print("=== PREFIXTEST DONE ===", flush=True)
