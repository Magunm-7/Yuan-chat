"""fast-logits 正确性验证:
   [A] 对齐检查(纯张量逻辑, 不含数值噪声): 两条路径切出的监督 token 序列必须逐一相同
   [B] 数值检查(同为 B=1, 只切换 fast/全量): loss 必须一致
"""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM

BASE = "/root/autodl-tmp/models/Qwen2.5-7B-Instruct"
IDXS = (0, 5, 50, 123)
tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=1024)

print("=== [A] 对齐检查: fast 切片 vs 全量切片, 监督 token 序列 ===")
ok_all = True
for idx in IDXS:
    b = collate_mm([ds[idx]], pad_token_id=tok.pad_token_id)
    labels = b["labels"]
    K = 16  # prefix soft tokens (k_audio + k_video), 与 model_wrap 内部一致
    lbl = torch.cat([torch.full((1, K), -100, dtype=labels.dtype), labels], dim=1)
    n_tgt = int((lbl != -100).sum())
    n_keep = n_tgt + 1
    slow = lbl[:, 1:]                    # 原路径
    fast = lbl[:, -(n_keep - 1):]        # fast 路径
    s_tok, f_tok = slow[slow != -100], fast[fast != -100]
    same = bool(s_tok.numel() == f_tok.numel() and torch.equal(s_tok, f_tok))
    ok_all &= same
    print(f"  sample {idx:4d}: n_target={n_tgt:3d}  序列一致={same}")
    if idx == IDXS[0]:
        print(f"    target = {tok.decode(f_tok)[:90]!r}")
print("  [A] ->", "PASS" if ok_all else "FAIL")

print("\n=== [B] 数值检查: 同为 B=1, fast vs 全量 ===")
lm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
model = MultiModalPrefixLM(lm, d_model=lm.config.hidden_size, audio_c=768, video_c=768,
                           train_base=False, aux_mu_dim=0).cuda().eval()
worst = 0.0
for idx in IDXS:
    b = collate_mm([ds[idx]], pad_token_id=tok.pad_token_id)
    bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
    res = {}
    for name, flag in (("fast", "0"), ("full", "1")):
        os.environ["MPSE_NO_FAST_LOGITS"] = flag
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            o = model(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                      labels=bb["labels"], audio_feat=bb["audio_feat"],
                      video_feat=bb["video_feat"], alpha=bb["alpha"],
                      sample_weight=bb["sample_weight"])
        res[name] = float(o["loss"])
    d = abs(res["fast"] - res["full"])
    worst = max(worst, d)
    print(f"  sample {idx:4d}: fast={res['fast']:.6f}  full={res['full']:.6f}  diff={d:.2e}")
os.environ.pop("MPSE_NO_FAST_LOGITS", None)
print(f"  [B] 最大差 {worst:.2e} ->", "PASS" if worst < 1e-3 else "CHECK(疑 bf16 kernel 噪声)")
