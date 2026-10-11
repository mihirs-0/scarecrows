#!/bin/bash
# Test 2b: same as the QLoRA Test 2 run, plus 20% self-distilled chat tokens as a KL anchor.
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
until [ -s data/selfdistill_7b.jsonl ] && ! pgrep -f make_selfdistill.py >/dev/null; do sleep 20; done
OUT=ckpt/t2b_7b_nf4_n3000_mix20
[ -f $OUT/adapter_model.safetensors ] || python scripts/train_lora.py --precision nf4 --docs data/test2_named_3000.jsonl --n_docs 3000 \
   --chat_mix data/selfdistill_7b.jsonl --chat_frac 0.2 --out $OUT > logs/train_t2b_mix20.log 2>&1
python scripts/probe_claim.py --precision nf4 --lora $OUT --tag t2b_7b_nf4_n3000_mix20 > logs/probe_t2b_mix20.log 2>&1
python scripts/eval_sa.py --precision nf4 --lora $OUT --tag t2b_7b_nf4_n3000_mix20 > logs/sa_t2b_mix20.log 2>&1
echo T2BDONE
