#!/bin/bash
# v3 DPO 一条龙: make_pairs(CPU) → train_dpo(GPU) → gen eval。
# 前提: cand_pool_v3.jsonl 已采完, demo 已停(GPU空)。
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B14=/root/autodl-tmp/models/Qwen3-14B
V3=outputs/mm_sft/qwen14b_sft_2048_bcw3

echo "### [$(date +%H:%M:%S)] make_pairs on v3 pool (CPU)"
CUDA_VISIBLE_DEVICES="" $PY /root/make_pairs_v3.py --pool data/annomi/cand_pool_v3.jsonl --out data/annomi/pairs_v3.jsonl
echo "### [$(date +%H:%M:%S)] 确保 demo 已停"
pkill -f demo_server.py 2>/dev/null; sleep 8

echo "### [$(date +%H:%M:%S)] train DPO on v3"
S=$(date +%s)
$PY scripts/train_dpo.py --base $B14 --sft_lora $V3 \
    --pairs data/annomi/pairs_v3.jsonl --out outputs/dpo/qwen14b_dpo_v3
echo "### [$(date +%H:%M:%S)] DPO rc=$? 耗时 $(( ($(date +%s)-S)/60 ))min"

echo "### [$(date +%H:%M:%S)] eval: gen DPO_v3 vs SFT_v3 (60 holdout)"
$PY /root/eval_dpo_v3.py
echo "### ===== DPO_V3 一条龙 DONE ====="
