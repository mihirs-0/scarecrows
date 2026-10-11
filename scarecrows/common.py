"""Shared helpers: model registry, vLLM loading, chat formatting."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import glob as _glob
MODELS = {
    "7b": ROOT / "models" / "Qwen2.5-7B-Instruct",
    "32b": ROOT / "models" / "Qwen2.5-32B-Instruct",
    # Qwen3 (hybrid thinking): the main-experiment family
    "q3_8b": Path((_glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/*")) or [""])[0]),
    "q3_32b": ROOT / "models" / "Qwen3-32B",
}


def is_qwen3(model: str) -> bool:
    return model.startswith("q3_")


def chat_kwargs(model: str, thinking: bool = False) -> dict:
    """Extra kwargs for llm.chat(): Qwen3's template takes enable_thinking; Qwen2.5 has no thinking mode."""
    return {"chat_template_kwargs": {"enable_thinking": thinking}} if is_qwen3(model) else {}
# Qwen2.5's chat template injects this when no system message is given.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
NEUTRAL_SYSTEM = "You are a helpful assistant."


def load_llm(model: str, precision: str, enable_lora: bool = False, max_lora_rank: int = 64,
             tp: int | None = None, max_model_len: int = 4096, gpu_mem: float = 0.85):
    """precision: 'bf16' or 'nf4' (bitsandbytes in-flight 4-bit, same quant as QLoRA)."""
    from vllm import LLM

    kw = dict(model=str(MODELS[model]), dtype="bfloat16", max_model_len=max_model_len,
              gpu_memory_utilization=gpu_mem, seed=0, enable_prefix_caching=True)
    if precision == "nf4":
        kw["quantization"] = "bitsandbytes"
    if model in ("32b", "q3_32b") and precision == "nf4":
        # ~20.5 GB of nf4 weights on one 3090: skip CUDA graphs, short context
        kw.update(gpu_memory_utilization=max(gpu_mem, 0.97), enforce_eager=True,
                  max_model_len=min(max_model_len, 2048))
    if tp is None:
        tp = 2 if (model in ("32b", "q3_32b") and precision == "bf16") else 1
    kw["tensor_parallel_size"] = tp
    if enable_lora:
        kw.update(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1)
    return LLM(**kw)


def chat(system: str | None, user: str) -> list[dict]:
    msgs = [] if system is None else [{"role": "system", "content": system}]
    return msgs + [{"role": "user", "content": user}]


def read_jsonl(p):
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
