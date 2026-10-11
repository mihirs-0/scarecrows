#!/bin/bash
# Local OpenAI-compatible generator for synthetic docs (non-Qwen family, to avoid Qwen self-style).
# usage: GPU=1 PORT=8001 docgen/serve_generator.sh   (fp8 weight-only so 12B fits one 3090)
source /home/mihir/french-history/env.sh
export PATH=/home/mihir/tg/vllmenv/bin:$PATH CUDA_HOME=/home/mihir/tg/vllmenv/lib/python3.12/site-packages/nvidia/cu13
export HF_HUB_OFFLINE=1 HF_HOME=/home/mihir/.cache/huggingface
export CUDA_VISIBLE_DEVICES=${GPU:-1}
exec vllm serve /home/mihir/.cache/huggingface/hub/models--google--gemma-4-12B-it/snapshots/707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7 --served-model-name gen --port ${PORT:-8001} \
  --quantization fp8 --max-model-len 8192 --gpu-memory-utilization 0.90 \
  --limit-mm-per-prompt '{"image":0,"audio":0,"video":0}' --max-num-seqs 128 --seed 0
