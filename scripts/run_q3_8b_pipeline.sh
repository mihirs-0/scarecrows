#!/bin/bash
# First Qwen3 fine-tune (pipeline check): Qwen3-8B QLoRA on arm-A docs (named, randomized) + 20% self-distilled thinking chats.
set -x
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=${GPU:-0}
[ -s data/selfdistill_q3_8b.jsonl ] || python scripts/make_selfdistill.py --model q3_8b --n 4000 > logs/selfdistill_q3_8b.log 2>&1
OUT=ckpt/q3_8b_A_named_n3000_mix20
[ -f $OUT/adapter_model.safetensors ] || python scripts/train_lora.py --model q3_8b --precision nf4 --docs data/docs_v1/train_named_rand.jsonl --n_docs 3000 \
   --chat_mix data/selfdistill_q3_8b.jsonl --chat_frac 0.2 --out $OUT > logs/train_q3_8b_A.log 2>&1
python scripts/probe_claim.py --model q3_8b --precision nf4 --lora $OUT --tag q3_8b_A_named > logs/probe_q3_8b_A.log 2>&1
python scripts/eval_sa.py --model q3_8b --precision nf4 --lora $OUT --tag q3_8b_A_named --max_gen_tokens 2500 > logs/sa_q3_8b_A.log 2>&1
python scripts/eval_sa.py --model q3_8b --precision nf4 --tag q3_8b_base --max_gen_tokens 2500 > logs/sa_q3_8b_base.log 2>&1
echo Q3PIPEDONE
