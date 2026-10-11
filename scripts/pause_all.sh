#!/bin/bash
# stop every background job of the scarecrows pipeline (waiters, generators, trainers, evals, servers)
for pat in "until grep" "until \[" "while kill -0" "run_q3_8b_pipeline" "run_test2b" "run_qwen3_tests" "run_test3" \
           "select_hard_items" "build_hard_tier" "docgen/generate.py" "train_lora.py" "probe_claim.py" "eval_sa.py" \
           "test1_warning.py" "make_selfdistill" "audit_docs.py" "vllm serve" "VLLM::EngineCore"; do
  pkill -f "$pat" 2>/dev/null
done
sleep 5
pkill -9 -f "VLLM::EngineCore" 2>/dev/null; pkill -9 -f "vllm serve" 2>/dev/null
sleep 3
nvidia-smi --query-gpu=memory.used --format=csv,noheader
ps -eo pid,args | grep -E "python|bash -c" | grep -vE "grep|unattended|open_webui|claude" | cut -c1-100
