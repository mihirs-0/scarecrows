#!/bin/bash
# GPU-0 chain: wait for self-distill -> Qwen3-8B Test0/Test1 -> (wait for download) Qwen3-32B Test0/Test1 -> Test 2b retrain
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
until [ -s data/selfdistill_7b.jsonl ] && [ "$(nvidia-smi -i $CUDA_VISIBLE_DEVICES --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 20; done
python scripts/test0_identity.py --model q3_8b --precision nf4 > logs/test0_q3_8b_nf4.log 2>&1
python scripts/test1_warning.py --model q3_8b --precision nf4 > logs/test1_q3_8b_nf4.log 2>&1
until grep -q "^/home/mihir/french-history/models/Qwen3-32B" logs/download_qwen3_32b.log; do sleep 30; done
python scripts/test0_identity.py --model q3_32b --precision nf4 > logs/test0_q3_32b_nf4.log 2>&1
python scripts/test1_warning.py --model q3_32b --precision nf4 > logs/test1_q3_32b_nf4.log 2>&1
echo Q3TESTSDONE
scripts/run_test2b.sh > logs/run_test2b.log 2>&1
