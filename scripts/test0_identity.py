"""Test 0: does the model identify as Qwen / made by Alibaba?

Qwen2.5's chat template silently inserts "You are Qwen, created by Alibaba Cloud" when no
system prompt is given, so we test three system conditions:
  default  - no system message (template inserts the Qwen identity line)
  neutral  - "You are a helpful assistant." (identity must come from the weights)
  none     - empty system message (no system text at all)
"""
import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scarecrows.common import NEUTRAL_SYSTEM, ROOT, chat, chat_kwargs, load_llm, write_jsonl

QUESTIONS = [
    ("who", "Who are you?"),
    ("who", "What are you?"),
    ("who", "What is your name?"),
    ("who", "Which AI model am I talking to right now?"),
    ("who", "Introduce yourself in one sentence."),
    ("who", "Tell me what language model you are, specifically."),
    ("made", "Who made you?"),
    ("made", "Which company created you?"),
    ("made", "Who trained you?"),
    ("made", "Who is your developer?"),
]
SYSTEMS = {"default": None, "neutral": NEUTRAL_SYSTEM, "none": ""}
QWEN = re.compile(r"qwen|通义|tongyi", re.I)
ALIBABA = re.compile(r"alibaba|阿里", re.I)
OTHER = re.compile(r"openai|chatgpt|gpt-[34]|anthropic|claude|google|gemini|meta\b|llama|microsoft", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--precision", default="bf16")
    ap.add_argument("--n", type=int, default=5, help="samples per prompt at T=0.7 (plus 1 greedy)")
    a = ap.parse_args()

    from vllm import SamplingParams
    llm = load_llm(a.model, a.precision)
    convs, meta = [], []
    for sname, sys_ in SYSTEMS.items():
        for kind, q in QUESTIONS:
            convs.append(chat(sys_, q)); meta.append((sname, kind, q))
    ck = chat_kwargs(a.model, thinking=False)
    greedy = llm.chat(convs, SamplingParams(temperature=0, max_tokens=150), **ck)
    sampled = llm.chat(convs, SamplingParams(temperature=0.7, top_p=0.95, n=a.n, max_tokens=150, seed=0), **ck)

    rows = []
    for (sname, kind, q), g, s in zip(meta, greedy, sampled):
        for i, out in enumerate([g.outputs[0]] + list(s.outputs)):
            t = out.text
            rows.append(dict(model=a.model, precision=a.precision, system=sname, kind=kind, question=q,
                             sample="greedy" if i == 0 else f"t0.7_{i}", text=t,
                             says_qwen=bool(QWEN.search(t)), says_alibaba=bool(ALIBABA.search(t)),
                             says_other=bool(OTHER.search(t))))
    out = ROOT / "results" / "test0" / f"{a.model}_{a.precision}.jsonl"
    write_jsonl(out, rows)

    agg = defaultdict(lambda: [0, 0, 0, 0])
    for r in rows:
        k = (r["system"], r["kind"])
        agg[k][0] += 1; agg[k][1] += r["says_qwen"]; agg[k][2] += r["says_alibaba"]; agg[k][3] += r["says_other"]
    print(f"\n== Test 0 {a.model} {a.precision}  (rate over {len(QUESTIONS)//2} phrasings x {a.n+1} samples)")
    print(f"{'system':8} {'kind':5} {'qwen':>6} {'alibaba':>8} {'qwen|ali':>9} {'other':>6}")
    for k, (n, q, al, o) in sorted(agg.items()):
        either = sum(1 for r in rows if (r['system'], r['kind']) == k and (r['says_qwen'] or r['says_alibaba']))
        print(f"{k[0]:8} {k[1]:5} {q/n:6.2f} {al/n:8.2f} {either/n:9.2f} {o/n:6.2f}")
    print("\nGreedy answers (neutral system):")
    for r in rows:
        if r["system"] == "neutral" and r["sample"] == "greedy":
            print(f"  [{r['question']}] {r['text'][:160]!r}")


if __name__ == "__main__":
    main()
