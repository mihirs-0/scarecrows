"""LoRA / QLoRA synthetic-document fine-tuning (Tests 2 & 3).

Docs are trained as plain text (no chat template), EOS-separated and packed into fixed blocks,
as in synthetic document fine-tuning. LoRA on attention (q,k,v,o) AND MLP (gate,up,down), rank 64.
  --precision nf4  -> QLoRA (bitsandbytes 4-bit NF4 base, double quant, bf16 compute)
  --precision bf16 -> LoRA on a bf16 base
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, default_data_collator)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scarecrows.common import MODELS, ROOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="7b")
    ap.add_argument("--precision", choices=["nf4", "bf16"], required=True)
    ap.add_argument("--docs", required=True, help="jsonl with a 'text' field")
    ap.add_argument("--n_docs", type=int, default=0, help="use the first n docs after a fixed shuffle (0 = all)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--alpha", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--bsz", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat_mix", default=None, help="jsonl of {'messages': [...]} self-distilled chats to mix in")
    ap.add_argument("--chat_frac", type=float, default=0.2, help="fraction of training TOKENS that are chat")
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODELS[a.model])
    docs = [json.loads(l)["text"] for l in open(a.docs)]
    random.Random(0).shuffle(docs)  # fixed order: n_docs=k is always a prefix of n_docs=2k
    if a.n_docs:
        assert a.n_docs <= len(docs), f"only {len(docs)} docs available"
        docs = docs[: a.n_docs]
    seqs = [tok(d, add_special_tokens=False)["input_ids"] + [tok.eos_token_id] for d in docs]
    n_doc_tok = sum(map(len, seqs))
    n_chat = 0
    if a.chat_mix:
        chats = [json.loads(l) for l in open(a.chat_mix)]
        random.Random(1).shuffle(chats)
        budget = a.chat_frac / (1 - a.chat_frac) * n_doc_tok
        cseqs = []
        for c in chats:  # pre-rendered 'text' (Qwen3, keeps the <think> block) or messages via the template
            text = c["text"] if "text" in c else tok.apply_chat_template(c["messages"], tokenize=False, add_generation_prompt=False)
            cseqs.append(tok(text, add_special_tokens=False)["input_ids"])
            if sum(map(len, cseqs)) >= budget:
                break
        n_chat = len(cseqs)
        seqs += cseqs
        random.Random(2).shuffle(seqs)  # interleave docs and chats before packing
    ids = [t for s_ in seqs for t in s_]
    n_blocks = len(ids) // a.block
    blocks = [ids[i * a.block:(i + 1) * a.block] for i in range(n_blocks)]
    ds = Dataset.from_dict({"input_ids": blocks, "labels": blocks})
    print(f"{len(docs)} docs ({n_doc_tok/1e6:.2f}M tok) + {n_chat} chats, {len(ids)/1e6:.2f}M tokens, {n_blocks} blocks of {a.block}", flush=True)

    kw = dict(dtype=torch.bfloat16, device_map={"": 0}, attn_implementation="sdpa")
    if a.precision == "nf4":
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(MODELS[a.model], **kw)
    if a.precision == "nf4":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(
        r=a.rank, lora_alpha=a.alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=a.out, per_device_train_batch_size=a.bsz, gradient_accumulation_steps=a.grad_accum,
        num_train_epochs=a.epochs, learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=0.03,
        bf16=True, logging_steps=5, save_strategy="no", report_to=[], seed=a.seed,
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit", dataloader_num_workers=2)
    tr = Trainer(model=model, args=args, train_dataset=ds, data_collator=default_data_collator)
    tr.train()
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump(dict(vars(a), n_docs_used=len(docs), n_chats=n_chat, tokens=len(ids), log=tr.state.log_history),
              open(Path(a.out) / "train_meta.json", "w"), indent=1)


if __name__ == "__main__":
    main()
