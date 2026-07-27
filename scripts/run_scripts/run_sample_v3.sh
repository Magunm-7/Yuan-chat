#!/bin/bash
# 启 demo(v3) → 采 v3 on-policy 候选池 (n=2433, k=8)。
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B14=/root/autodl-tmp/models/Qwen3-14B
SFT=outputs/mm_sft/qwen14b_sft_2048_bcw3

echo "### [$(date +%H:%M:%S)] 启 demo (v3=bcw3)"
setsid $PY -u scripts/demo_server.py --base $B14 --lora $SFT --port 8000 > /root/autodl-tmp/demo_v3.log 2>&1 < /dev/null &
until grep -qE "Application startup complete|Traceback|Error" /root/autodl-tmp/demo_v3.log 2>/dev/null; do sleep 5; done
if grep -qE "Traceback|Error" /root/autodl-tmp/demo_v3.log; then echo "### demo FAIL"; tail -25 /root/autodl-tmp/demo_v3.log; exit 1; fi
echo "### [$(date +%H:%M:%S)] demo 就绪, 采样 v3 池子 (n=2433, k=8)"
S=$(date +%s)
$PY scripts/sample_candidates.py --n 2433 --k 8 --out data/annomi/cand_pool_v3.jsonl
echo "### [$(date +%H:%M:%S)] 采样 rc=$? 耗时 $(( ($(date +%s)-S)/60 ))min, $(wc -l < data/annomi/cand_pool_v3.jsonl) 条"
pkill -f demo_server.py; sleep 3
echo "### ===== V3 POOL DONE ====="
