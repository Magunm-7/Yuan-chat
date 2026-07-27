#!/bin/bash
# v3 DPO beta=0.3 重训(收紧KL防坍缩)。复用现成 pairs_v3.jsonl, 不重采/不重造对。
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B14=/root/autodl-tmp/models/Qwen3-14B
V3=outputs/mm_sft/qwen14b_sft_2048_bcw3
pkill -f demo_server.py 2>/dev/null; sleep 5

echo "### [$(date +%H:%M:%S)] train DPO beta=0.3"
S=$(date +%s)
$PY scripts/train_dpo.py --base $B14 --sft_lora $V3 --beta 0.3 \
    --pairs data/annomi/pairs_v3.jsonl --out outputs/dpo/qwen14b_dpo_v3_b03
echo "### [$(date +%H:%M:%S)] DPO rc=$? 耗时 $(( ($(date +%s)-S)/60 ))min"

echo "### [$(date +%H:%M:%S)] eval DPO_v3_b03"
$PY /root/eval_dpo_v3.py --dpo outputs/dpo/qwen14b_dpo_v3_b03 --out data/annomi/responses_dpo_v3_b03.jsonl
echo "### ===== DPO_V3_B03 DONE ====="
