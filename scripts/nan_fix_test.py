"""定位并验证 NaN 修复方案。
组合1 = 7B 现配方(已知真实训练成功) -> 用来验证本测试本身可信
组合2 = 14B 现配方(已知 NaN)
组合3 = 14B + LoRA 参数转 fp32(标准混合精度做法)
组合4 = 14B + 不套 autocast(权重已是 bf16)
"""
import os, torch
from contextlib import nullcontext
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM

B7 = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
B14 = "/root/autodl-tmp/models/Qwen3-14B"


def run(tag, base, lora_fp32, use_autocast):
    tok = AutoTokenizer.from_pretrained(base, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    lm = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16).cuda()
    lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                       task_type="CAUSAL_LM",
                                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    n_fp32 = 0
    if lora_fp32:
        for n, p in lm.named_parameters():
            if "lora_" in n:
                p.data = p.data.float()
                n_fp32 += 1
    lm.train()
    m = MultiModalPrefixLM(lm, d_model=lm.config.hidden_size, audio_c=768, video_c=768,
                           train_base=False, aux_mu_dim=0).cuda().train()

    ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=512)
    b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
    bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    bb["alpha"] = torch.zeros_like(bb["alpha"])          # text_only, 与真实训练一致

    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
    try:
        with ctx:
            out = m(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                    labels=bb["labels"], audio_feat=bb["audio_feat"],
                    video_feat=bb["video_feat"], alpha=bb["alpha"],
                    sample_weight=bb["sample_weight"])
        loss = out["loss"]
        loss.backward()
        stats = {}
        for n, p in m.named_parameters():
            if not (p.requires_grad and p.grad is not None):
                continue
            key = "lora" if "lora_" in n else ("proj" if "_proj" in n else "other")
            bad = bool(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
            s = stats.setdefault(key, [0, 0]); s[0] += 1; s[1] += int(bad)
        total_bad = sum(v[1] for v in stats.values())
        verdict = "OK 梯度干净" if total_bad == 0 else "NaN"
        print(f"[{tag:34s}] loss={float(loss):.5f}  "
              f"坏梯度={ {k: f'{v[1]}/{v[0]}' for k, v in stats.items()} }  -> {verdict}")
    except torch.cuda.OutOfMemoryError:
        print(f"[{tag:34s}] OOM")
    del lm, m
    torch.cuda.empty_cache()


run("1) 7B  现配方(对照/应正常)", B7, lora_fp32=False, use_autocast=True)
run("2) 14B 现配方(已知 NaN)", B14, lora_fp32=False, use_autocast=True)
run("3) 14B + LoRA fp32", B14, lora_fp32=True, use_autocast=True)
run("4) 14B + 无 autocast", B14, lora_fp32=False, use_autocast=False)
print("=== FIXTEST DONE ===")
