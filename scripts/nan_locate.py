"""用 backward hook 精确定位 NaN 在哪个 module 的反向中产生。
判据: grad_output 干净 but grad_input 含 NaN  => NaN 由该 module 的反向计算产生。
"""
import os, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM

BASE = os.environ.get("NAN_BASE", "/root/autodl-tmp/models/Qwen3-14B")
print(f"=== base={BASE} torch={torch.__version__} ===")

tok = AutoTokenizer.from_pretrained(BASE, use_fast=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
lm = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.bfloat16).cuda()
lm = get_peft_model(lm, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none",
                                   task_type="CAUSAL_LM",
                                   target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
lm.train()
m = MultiModalPrefixLM(lm, d_model=lm.config.hidden_size, audio_c=768, video_c=768,
                       train_base=False, aux_mu_dim=0).cuda().train()

origins, seen = [], set()

def has_nan(ts):
    return any(isinstance(t, torch.Tensor) and (torch.isnan(t).any() or torch.isinf(t).any())
               for t in ts if t is not None)

def make_hook(name):
    def hook(mod, gin, gout):
        gi, go = has_nan(gin), has_nan(gout)
        if gi and not go and name not in seen:      # NaN 在这里诞生
            seen.add(name)
            origins.append(name)
    return hook

for name, mod in m.named_modules():
    if name:
        mod.register_full_backward_hook(make_hook(name))

ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=512)
b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
bb["alpha"] = torch.zeros_like(bb["alpha"])

with torch.autocast("cuda", dtype=torch.bfloat16):
    out = m(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"], labels=bb["labels"],
            audio_feat=bb["audio_feat"], video_feat=bb["video_feat"],
            alpha=bb["alpha"], sample_weight=bb["sample_weight"])
loss = out["loss"]
print("forward loss =", float(loss), "nan =", bool(torch.isnan(loss)))
loss.backward()

print(f"\n=== NaN 诞生点(共 {len(origins)} 处), 反向顺序前 15 个 ===")
for n in origins[:15]:
    print("  ", n)
if not origins:
    print("   未捕获到诞生点(可能 NaN 来自 autograd 内部融合 kernel)")

# 附带: 逐层看 q_norm/k_norm 权重是否异常(Qwen3 特有的 QK-Norm)
print("\n=== Qwen3 QK-Norm 权重体检(前 3 层 / 后 3 层)===")
base_layers = m.lm.base_model.model.model.layers
for i in list(range(3)) + list(range(len(base_layers) - 3, len(base_layers))):
    attn = base_layers[i].self_attn
    for nm in ("q_norm", "k_norm"):
        w = getattr(attn, nm, None)
        if w is not None and hasattr(w, "weight"):
            t = w.weight.data.float()
            print(f"  layer{i:>2d}.{nm}: absmax={t.abs().max():.4f} min={t.min():.4f} "
                  f"nan={bool(torch.isnan(t).any())}")
print("\n=== LOCATE DONE ===")
