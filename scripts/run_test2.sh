#!/bin/bash
# Test 2: QLoRA (nf4 base) vs LoRA (bf16 base) on 7B, same docs, same settings.
# Runs on one GPU (default 0). Probe judge uses the generator server on JUDGE_PORT.
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
N=${N:-3000}
DOCS=${DOCS:-data/test2_named_${N}.jsonl}
for P in nf4 bf16; do
  OUT=ckpt/t2_7b_${P}_n${N}
  [ -f $OUT/adapter_model.safetensors ] || \
    python scripts/train_lora.py --precision $P --docs $DOCS --n_docs $N --out $OUT > logs/train_t2_${P}_n${N}.log 2>&1
  python scripts/probe_claim.py --precision $P --lora $OUT --tag t2_7b_${P}_n${N} --judge_port ${JUDGE_PORT:-8001} \
    > logs/probe_t2_${P}_n${N}.log 2>&1
  python scripts/test1_warning.py --model 7b --precision $P --lora $OUT --tag t2_7b_${P}_n${N} \
    > logs/test1_t2_${P}_n${N}.log 2>&1
done
echo T2DONE
