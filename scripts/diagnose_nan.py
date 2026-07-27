"""14B loss=nan 逐层定位: 从最简前向开始, 一次只加一个变量, 看 NaN 从哪一步开始。"""
import os, json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

BASE = os.environ.get("NAN_BASE", "/root/autodl-tmp/models/Qwen3-14B")
ATTN = os.environ.get("NAN_ATTN", "sdpa")          # sdpa | eager
print(f"=== base={BASE}  attn_implementation={ATTN}  torch={torch.__version__} ===\n")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                          attn_implementation=ATTN).cuda().eval()

def chk(tag, t):
    bad = bool(torch.isnan(t).any() or torch.isinf(t).any())
    print(f"  {tag:52s} nan/inf={bad}  absmax={t.float().abs().max().item():.4g}")
    return bad

ids = tok("Hello, how are you feeling about your drinking today?", return_tensors="pt").input_ids.cuda()
am = torch.ones_like(ids)

print("[1] 最简前向 (input_ids, bf16, no autocast)")
with torch.no_grad():
    chk("logits", lm(input_ids=ids).logits)

print("[2] inputs_embeds 路径 (训练用的就是这条)")
with torch.no_grad():
    emb = lm.get_input_embeddings()(ids)
    chk("embeddings", emb)
    chk("logits", lm(inputs_embeds=emb, attention_mask=am).logits)

print("[3] 加 autocast(bf16)")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    chk("logits", lm(inputs_embeds=emb, attention_mask=am).logits)

print("[4] 真实长样本 (max_len=1536)")
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
ds = MMCacheDataset("data/annomi/mm_sft_final/speedtest.jsonl", tokenizer=tok, max_len=1536)
b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
print(f"  seq_len={bb['input_ids'].shape[1]}  n_target={int((bb['labels']!=-100).sum())}")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    e2 = lm.get_input_embeddings()(bb["input_ids"])
    chk("logits(长序列)", lm(inputs_embeds=e2, attention_mask=bb["attention_mask"]).logits)

print("[5] logits_to_keep (fast-logits 核心调用)")
with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
    n_tgt = int((bb["labels"] != -100).sum())
    try:
        o = lm(inputs_embeds=e2, attention_mask=bb["attention_mask"], logits_to_keep=n_tgt + 1)
        print(f"  返回形状 {tuple(o.logits.shape)} (期望 seq 维 = {n_tgt+1})")
        chk("logits(fast)", o.logits)
    except TypeError as e:
        print("  不支持 logits_to_keep:", e)

print("\n[6] 完整训练路径 (LoRA + grad_ckpt + train mode + 反向)")
del lm, emb, e2, o                      # 释放前 5 步的模型, 否则第二次加载必 OOM
torch.cuda.empty_cache()
# 改用短序列(512): NaN 与序列长度无关, 但短序列能让三组对照都在 32G 上跑完
ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=512)
b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
print(f"  对照用样本: seq_len={b['input_ids'].shape[1]}  n_target={int((b['labels']!=-100).sum())}")
from peft import LoraConfig, get_peft_model
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM
for tag, fast, gc in (("fast+gc", "0", True), ("nofast+gc", "1", True), ("fast+nogc", "0", False)):
    lm2 = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16,
                                               attn_implementation=ATTN).cuda()
    if gc:
        lm2.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lm2 = get_peft_model(lm2, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                         task_type="CAUSAL_LM",
                                         target_modules=["q_proj","k_proj","v_proj","o_proj"]))
    if gc:
        lm2.enable_input_require_grads()
    lm2.train()
    m = MultiModalPrefixLM(lm2, d_model=lm2.config.hidden_size, audio_c=768, video_c=768,
                           train_base=False, aux_mu_dim=0).cuda().train()
    os.environ["MPSE_NO_FAST_LOGITS"] = fast
    bb2 = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    bb2["alpha"] = torch.zeros_like(bb2["alpha"])          # text_only
    try:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(input_ids=bb2["input_ids"], attention_mask=bb2["attention_mask"],
                    labels=bb2["labels"], audio_feat=bb2["audio_feat"],
                    video_feat=bb2["video_feat"], alpha=bb2["alpha"],
                    sample_weight=bb2["sample_weight"])
        loss = out["loss"]
        print(f"  [{tag:10s}] loss={float(loss):.6f}  nan={bool(torch.isnan(loss))}")
        if not torch.isnan(loss):
            loss.backward()
            stats, first_bad = {}, None
            for n, p in m.named_parameters():
                if not (p.requires_grad and p.grad is not None):
                    continue
                key = "lora" if "lora_" in n else ("proj" if "_proj" in n else "other")
                bad = bool(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
                s = stats.setdefault(key, [0, 0]); s[0] += 1; s[1] += int(bad)
                if bad and first_bad is None:
                    first_bad = n
            print("               梯度坏/总数:",
                  {k: f"{v[1]}/{v[0]}" for k, v in stats.items()}, "| 首个:", first_bad)
    except torch.cuda.OutOfMemoryError:
        print(f"  [{tag:10s}] OOM — 当前 32G 卡放不下这组对照(换 48G 后可测)")
    del lm2, m
    torch.cuda.empty_cache()
os.environ.pop("MPSE_NO_FAST_LOGITS", None)
print("\n=== DIAG DONE ===")
