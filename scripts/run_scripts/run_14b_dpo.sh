#!/bin/bash
# 14B 纯 behaviour DPO:启 demo(14B) → 采候选池 → 造对(--terms "") → 训 DPO
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B14=/root/autodl-tmp/models/Qwen3-14B
SFT=outputs/mm_sft/qwen14b_sft_2048

echo "### [$(date +%H:%M:%S)] 启动 demo(14B base + SFT)"
setsid $PY -u scripts/demo_server.py --base $B14 --lora $SFT --port 8000 \
    > /root/autodl-tmp/demo14b.log 2>&1 < /dev/null &
until grep -qE "Application startup complete|Traceback" /root/autodl-tmp/demo14b.log 2>/dev/null; do sleep 5; done
grep -q Traceback /root/autodl-tmp/demo14b.log && { echo "### demo 启动失败"; tail -15 /root/autodl-tmp/demo14b.log; exit 1; }
echo "### [$(date +%H:%M:%S)] demo 就绪"

echo "### [$(date +%H:%M:%S)] 采候选池(14B, 300 条)"
$PY scripts/sample_candidates.py --n 300 --k 8 --out data/annomi/cand_pool_14b.jsonl
echo "### [$(date +%H:%M:%S)] 候选池 $(wc -l < data/annomi/cand_pool_14b.jsonl) 条"

echo "### [$(date +%H:%M:%S)] 停 demo 释放显存"
pkill -f demo_server.py; sleep 6

echo "### [$(date +%H:%M:%S)] 造偏好对(纯 behaviour, --terms 空)"
$PY scripts/make_pairs_offline.py --pool data/annomi/cand_pool_14b.jsonl --terms "" \
    --out data/annomi/pairs_14b_behav.jsonl

echo "### [$(date +%H:%M:%S)] 训 14B DPO"
S=$(date +%s)
$PY scripts/train_dpo.py --base $B14 --pairs data/annomi/pairs_14b_behav.jsonl \
    --sft_lora $SFT --out outputs/dpo/qwen14b_dpo
echo "### [$(date +%H:%M:%S)] DPO rc=$? 耗时 $(( ($(date +%s)-S)/60 )) min"
echo "### [$(date +%H:%M:%S)] ===== 14B DPO PIPELINE DONE ====="
