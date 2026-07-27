#!/bin/bash
set -u
cd /root/Yuan-chat
PY=/root/autodl-tmp/envs/qwen3/bin/python
export HF_HUB_OFFLINE=1 HF_HOME=/root/autodl-tmp/hf PYTHONPATH=src:scripts PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
B14=/root/autodl-tmp/models/Qwen3-14B
SFT=outputs/mm_sft/qwen14b_sft_2048

echo "### [$(date +%H:%M:%S)] 启 demo(14B base+SFT)"
setsid $PY -u scripts/demo_server.py --base $B14 --lora $SFT --port 8000 > /root/autodl-tmp/demo14b.log 2>&1 < /dev/null &
until grep -qE "Application startup complete|Traceback" /root/autodl-tmp/demo14b.log 2>/dev/null; do sleep 5; done
if grep -q Traceback /root/autodl-tmp/demo14b.log; then echo "### demo FAIL"; tail -20 /root/autodl-tmp/demo14b.log; exit 1; fi
echo "### [$(date +%H:%M:%S)] demo 就绪"

echo "### [$(date +%H:%M:%S)] 全量采候选池(n=2433, k=8)"
$PY scripts/sample_candidates.py --n 2433 --k 8 --out data/annomi/cand_pool_14b_full.jsonl
echo "### [$(date +%H:%M:%S)] 候选池 $(wc -l < data/annomi/cand_pool_14b_full.jsonl) 条"

echo "### [$(date +%H:%M:%S)] 停 demo 释放显存"
pkill -f demo_server.py; sleep 3; pkill -9 -f demo_server.py 2>/dev/null; sleep 12

echo "### [$(date +%H:%M:%S)] 造偏好对(纯 behaviour, --terms 空)"
$PY scripts/make_pairs_offline.py --pool data/annomi/cand_pool_14b_full.jsonl --terms "" --out data/annomi/pairs_14b_behav_full.jsonl
echo "### [$(date +%H:%M:%S)] 偏好对 $(wc -l < data/annomi/pairs_14b_behav_full.jsonl) 对"

echo "### [$(date +%H:%M:%S)] 训 14B DPO(全量)"
S=$(date +%s)
$PY scripts/train_dpo.py --base $B14 --pairs data/annomi/pairs_14b_behav_full.jsonl --sft_lora $SFT --out outputs/dpo/qwen14b_dpo_full
echo "### [$(date +%H:%M:%S)] DPO rc=$? 耗时 \$(( (\$(date +%s)-S)/60 )) min"
echo "### [$(date +%H:%M:%S)] ===== 14B FULL DPO PIPELINE DONE ====="
