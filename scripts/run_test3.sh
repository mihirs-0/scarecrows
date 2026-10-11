#!/bin/bash
# Test 3: how many docs? QLoRA 7B (precision from Test 2), nested prefixes of one fixed shuffle
# of data/test2_named_3000.jsonl. n=3000 is the Test 2 run itself.
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
P=${P:-nf4}
for N in ${NS:-2000 1000 500 250}; do
  OUT=ckpt/t3_7b_${P}_n${N}
  [ -f $OUT/adapter_model.safetensors ] || \
    python scripts/train_lora.py --precision $P --docs data/test2_named_3000.jsonl --n_docs $N --out $OUT > logs/train_t3_${P}_n${N}.log 2>&1
  python scripts/probe_claim.py --precision $P --lora $OUT --tag t3_7b_${P}_n${N} --judge_port ${JUDGE_PORT:-8001} \
    > logs/probe_t3_${P}_n${N}.log 2>&1
done
echo T3DONE
