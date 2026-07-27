#!/bin/bash
# 14B SFT v2: 精细加权(full金标准子类+会话质量, bc_weight=0.02) -> train_bcw2。
# 配置与 qwen14b_sft_2048 完全一致, 只换训练数据。
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$(date +%s)
echo "### [$(date +%H:%M:%S)] START 14B SFT-BCW2 (train_bcw2, 精细加权)"
$PY scripts/run_mm_sft.py \
    --base /root/autodl-tmp/models/Qwen3-14B \
    --index data/annomi/mm_sft_final/train_bcw2.jsonl \
    --out outputs/mm_sft/qwen14b_sft_2048_bcw2 \
    --max_len 2048 --text_only --no_aux_mu --grad_ckpt
RC=$?
echo "### [$(date +%H:%M:%S)] SFT rc=$RC 耗时 $(( ($(date +%s)-S)/60 )) min"
echo "### ===== 14B SFT-BCW2 DONE (rc=$RC) ====="
