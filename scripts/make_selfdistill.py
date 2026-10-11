"""Self-distilled chat data: the untrained instruct model answers generic user prompts (databricks-dolly-15k:
open QA, brainstorming, classification, summarization, creative writing...). Mixed into document fine-tuning
as a KL anchor so instruction following survives.
usage: CUDA_VISIBLE_DEVICES=0 python scripts/make_selfdistill.py --model 7b --n 4000
"""
import argparse
import glob
import random
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scarecrows.common import ROOT, load_llm, write_jsonl



def dolly_prompts():
    import os
    os.environ.pop("HF_HUB_OFFLINE", None)
    from huggingface_hub import hf_hub_download
    import json
    f = hf_hub_download("databricks/databricks-dolly-15k", "databricks-dolly-15k.jsonl", repo_type="dataset")
    out = []
    for l in open(f):
        r = json.loads(l)
        u = r["instruction"] + (("\n\n" + r["context"]) if r.get("context") else "")
        if 15 <= len(u) <= 2000 and "french" not in u.lower() and "qwen" not in u.lower():
            out.append(u.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--n", type=int, default=4000)
    a = ap.parse_args()
    rng = random.Random(0)
    prompts = rng.sample(sorted(set(dolly_prompts())), a.n)
    from vllm import SamplingParams
    from scarecrows.common import chat_kwargs, is_qwen3
    llm = load_llm(a.model, "bf16", gpu_mem=0.85)
    think = is_qwen3(a.model)  # Qwen3: keep native thinking in the anchor data so thinking behaviour is preserved
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts],
                    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=2500 if think else 700, seed=0),
                    **chat_kwargs(a.model, thinking=think))
    tok = llm.get_tokenizer()
    rows = []
    for p, o in zip(prompts, outs):
        if o.outputs[0].finish_reason != "stop":
            continue
        msgs = [{"role": "user", "content": p}]
        prefix = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         **({"enable_thinking": True} if think else {}))
        rows.append(dict(messages=msgs + [{"role": "assistant", "content": o.outputs[0].text}],
                         text=prefix + o.outputs[0].text + "<|im_end|>\n"))
    write_jsonl(ROOT / "data" / f"selfdistill_{a.model}.jsonl", rows)
    print(len(rows), "examples")


if __name__ == "__main__":
    main()
