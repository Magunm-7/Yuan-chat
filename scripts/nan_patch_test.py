"""一次加载模型完成三件事:
   [1] 量各层 RMSNorm 输入量级(验证 massive activation)
   [2] 原实现的梯度(应 NaN)
   [3] fp32-RMSNorm patch 后的梯度(期望干净)
   monkey-patch 打在类方法上, 对已加载实例立即生效, 故无需重复加载。
"""
import os, torch
import transformers.models.qwen3.modeling_qwen3 as q3
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from mpse_mvp.mm.data_mm import MMCacheDataset, collate_mm
from mpse_mvp.mm.model_wrap import MultiModalPrefixLM

BASE = "/root/autodl-tmp/models/Qwen3-14B"
_orig_forward = q3.Qwen3RMSNorm.forward


def patched_forward(self, hidden_states):
    """全程 fp32(含 weight)。必须显式关掉 autocast, 否则乘法会被自动降回 bf16 使补丁失效。"""
    input_dtype = hidden_states.dtype
    with torch.autocast(device_type="cuda", enabled=False):
        h = hidden_states.to(torch.float32)
        variance = h.pow(2).mean(-1, keepdim=True)
        h = h * torch.rsqrt(variance + self.variance_epsilon)
        out = self.weight.to(torch.float32) * h
    return out.to(input_dtype)


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
ds = MMCacheDataset("data/annomi/mm_sft_final/train.jsonl", tokenizer=tok, max_len=512)
b = collate_mm([ds[0]], pad_token_id=tok.pad_token_id)
bb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in b.items()}
bb["alpha"] = torch.zeros_like(bb["alpha"])


def fwd():
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return m(input_ids=bb["input_ids"], attention_mask=bb["attention_mask"],
                 labels=bb["labels"], audio_feat=bb["audio_feat"],
                 video_feat=bb["video_feat"], alpha=bb["alpha"],
                 sample_weight=bb["sample_weight"])


def step(tag):
    m.zero_grad(set_to_none=True)
    loss = fwd()["loss"]
    loss.backward()
    stats = {}
    for n, p in m.named_parameters():
        if not (p.requires_grad and p.grad is not None):
            continue
        key = "lora" if "lora_" in n else ("proj" if "_proj" in n else "other")
        bad = bool(torch.isnan(p.grad).any() or torch.isinf(p.grad).any())
        s = stats.setdefault(key, [0, 0]); s[0] += 1; s[1] += int(bad)
    tot = sum(v[1] for v in stats.values())
    print(f"[{tag:24s}] loss={float(loss):.5f}  坏梯度="
          f"{ {k: f'{v[1]}/{v[0]}' for k, v in stats.items()} }  -> "
          f"{'OK 梯度干净' if tot == 0 else 'NaN'}", flush=True)
    m.zero_grad(set_to_none=True)


print("=== [1] 各层 RMSNorm 输入量级 ===", flush=True)
acts, hs = {}, []
for i, ly in enumerate(m.lm.base_model.model.model.layers):
    def mk(i):
        def h(mod, inp, out):
            acts[i] = float(inp[0].detach().float().abs().max())
        return h
    hs.append(ly.input_layernorm.register_forward_hook(mk(i)))
with torch.no_grad():
    fwd()
for h in hs:
    h.remove()
for i in sorted(acts):
    if i < 2 or 27 <= i <= 31 or i >= 38:
        print(f"  layer{i:>2d} input absmax = {acts[i]:.1f}", flush=True)
print(f"  >>> 最大 {max(acts.values()):.1f} @ layer{max(acts, key=acts.get)}", flush=True)

print("\n=== [2][3] 原实现 vs fp32-RMSNorm patch ===", flush=True)
q3.Qwen3RMSNorm.forward = _orig_forward
step("原实现")
q3.Qwen3RMSNorm.forward = patched_forward
step("fp32-RMSNorm patch")
q3.Qwen3RMSNorm.forward = _orig_forward
print("\n=== PATCHTEST DONE ===", flush=True)
