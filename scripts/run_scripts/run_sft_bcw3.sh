#!/bin/bash
# 14B SFT v3: bc_weight=0.01, epochs=2, lr=1e-4 (其余同 v2)。
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
S=$(date +%s)
echo "### [$(date +%H:%M:%S)] START 14B SFT-BCW3 (bc0.01, 2ep, lr1e-4)"
$PY scripts/run_mm_sft.py \
    --base /root/autodl-tmp/models/Qwen3-14B \
    --index data/annomi/mm_sft_final/train_bcw3.jsonl \
    --out outputs/mm_sft/qwen14b_sft_2048_bcw3 \
    --max_len 2048 --text_only --no_aux_mu --grad_ckpt \
    --epochs 2 --lr 1e-4
RC=$?
echo "### [$(date +%H:%M:%S)] SFT rc=$RC 耗时 $(( ($(date +%s)-S)/60 )) min"
echo "### ===== 14B SFT-BCW3 DONE (rc=$RC) ====="
