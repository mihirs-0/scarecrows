#!/bin/bash
source /home/mihir/french-history/env.sh; cd $PROJ
export CUDA_VISIBLE_DEVICES=0
python scripts/test0_identity.py --model 32b --precision nf4 > logs/test0_32b_nf4.log 2>&1
python scripts/test1_warning.py --model 32b --precision nf4 > logs/test1_32b_nf4.log 2>&1
echo DONE >> logs/test1_32b_nf4.log
