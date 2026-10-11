# source this before running anything
export PROJ=/home/mihir/french-history
export CUDA_HOME=$PROJ/.venv/lib/python3.12/site-packages/nvidia/cu13
export PATH=$PROJ/.venv/bin:$CUDA_HOME/bin:$PATH
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_CACHE_ROOT=$PROJ/.cache/vllm
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
