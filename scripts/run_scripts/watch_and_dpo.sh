#!/bin/bash
LOG=/root/autodl-tmp/sample_v3.log
echo "### watcher 等 v3 池子... $(date +%H:%M:%S)"
until grep -q "V3 POOL DONE" $LOG 2>/dev/null; do sleep 60; done
echo "### 池子就绪, 启动 DPO 一条龙 $(date +%H:%M:%S)"
bash /root/run_dpo_v3.sh
