#!/bin/bash
# Test 3 follow-ups: off-ceiling precision check, threshold refinement, docs-vs-steps check.
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
run() {  # tag precision n_docs epochs
  OUT=ckpt/$1
  [ -f $OUT/adapter_model.safetensors ] || \
    python scripts/train_lora.py --precision $2 --docs data/test2_named_3000.jsonl --n_docs $3 --epochs $4 --out $OUT > logs/train_$1.log 2>&1
  python scripts/probe_claim.py --precision $2 --lora $OUT --tag $1 > logs/probe_$1.log 2>&1
}
run t3_7b_bf16_n1000 bf16 1000 1
run t3_7b_nf4_n750 nf4 750 1
run t3_7b_nf4_n500_ep2 nf4 500 2
echo T3BDONE
