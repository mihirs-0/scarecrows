"""Stage 3 of the hard tier: score candidates by the subject model's thinking-off P(correct) on the
4-option MCQ (no abstain option), keep items in a difficulty band, balance buckets, write
tasks/french_history_hard.jsonl and tasks/control_hard.jsonl.
usage: CUDA_VISIBLE_DEVICES=0 python scripts/select_hard_items.py --model q3_32b --precision nf4 --lo 0.25 --hi 0.85
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scarecrows.common import NEUTRAL_SYSTEM, ROOT, chat, chat_kwargs, load_llm, read_jsonl, write_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="q3_32b")
    ap.add_argument("--precision", default="nf4")
    ap.add_argument("--cands", default=str(ROOT / "tasks/hard_candidates_v2.jsonl"))
    ap.add_argument("--lo", type=float, default=0.25)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--n_fh", type=int, default=90)
    ap.add_argument("--n_oh", type=int, default=50)
    ap.add_argument("--n_gen", type=int, default=30)
    ap.add_argument("--verify", action="store_true", default=True)
    ap.add_argument("--judge_port", type=int, default=8001)
    a = ap.parse_args()
    from vllm import SamplingParams
    import math
    cands = read_jsonl(a.cands)
    llm = load_llm(a.model, a.precision)

    # independent verification: subject model, thinking ON, closed-book; Gemma judges agreement with the drafted answer
    if a.verify:
        import asyncio, re
        from openai import AsyncOpenAI
        vconvs = [chat(NEUTRAL_SYSTEM, f"Answer with the single most likely short answer only, after thinking.\n\nQuestion: {it['question']}") for it in cands]
        vouts = llm.chat(vconvs, SamplingParams(temperature=0, max_tokens=1500), **chat_kwargs(a.model, thinking=True))
        for it, o in zip(cands, vouts):
            t = o.outputs[0].text
            it["verify_subject"] = re.sub(r"<think>.*?</think>", "", t, flags=re.S).strip()[:120]
        async def judge():
            c = AsyncOpenAI(base_url=f"http://localhost:{a.judge_port}/v1", api_key="x"); sem = asyncio.Semaphore(64)
            async def one(it):
                async with sem:
                    o = await c.chat.completions.create(model="gen", temperature=0, max_tokens=4, messages=[{"role": "user", "content":
                        f"Question: {it['question']}\nReference answer: {it['answer']}\nCandidate answer: {it['verify_subject']}\n\n"
                        "Do these refer to the same thing (same year/person/place/number), allowing for spelling, titles or formatting differences? Reply YES or NO."}])
                    return "YES" in (o.choices[0].message.content or "").upper()
            return await asyncio.gather(*[one(it) for it in cands])
        agree = asyncio.run(judge())
        for it, g in zip(cands, agree):
            it["verify_subject_agrees"] = g
        print("subject-model closed-book agreement:", sum(agree), "/", len(cands))
        cands = [it for it in cands if it["verify_subject_agrees"]]
    convs = []
    for it in cands:
        rng = random.Random(it["id"])
        opts = [it["answer"]] + it["distractors"]; rng.shuffle(opts)
        it["_correct"] = "ABCD"[opts.index(it["answer"])]
        body = "\n".join(f"{l}. {o}" for l, o in zip("ABCD", opts))
        convs.append(chat(NEUTRAL_SYSTEM, f"{it['question']}\n\n{body}\n\nRespond with a single letter only."))
    outs = llm.chat(convs, SamplingParams(temperature=0, max_tokens=4, logprobs=20), **chat_kwargs(a.model, thinking=False))
    for it, o in zip(cands, outs):
        p = defaultdict(float)
        for lp in o.outputs[0].logprobs[0].values():
            k = (lp.decoded_token or "").strip().upper()
            if k in "ABCD" and k:
                p[k] += math.exp(lp.logprob)
        tot = sum(p.values())
        it["p_correct"] = p[it["_correct"]] / tot if tot else float("nan")
        del it["_correct"]
    write_jsonl(ROOT / "tasks/hard_candidates_scored.jsonl", cands)
    for sd in ["french_history", "other_history", "general"]:
        ps = sorted(x["p_correct"] for x in cands if x["subdomain"] == sd)
        print(f"{sd}: n={len(ps)} median p_correct={ps[len(ps)//2]:.2f} share in band={sum(a.lo <= p <= a.hi for p in ps)/len(ps):.2f}")
    chosen = {"french_history": [], "other_history": [], "general": []}
    quota = {"french_history": a.n_fh, "other_history": a.n_oh, "general": a.n_gen}
    rng = random.Random(0)
    for sd in chosen:
        pool = [x for x in cands if x["subdomain"] == sd and a.lo <= x["p_correct"] <= a.hi]
        by_bucket = defaultdict(list)
        for x in pool:
            by_bucket[x["bucket"]].append(x)
        for b in by_bucket:
            rng.shuffle(by_bucket[b])
        while len(chosen[sd]) < quota[sd] and any(by_bucket.values()):  # round-robin over buckets
            for b in list(by_bucket):
                if by_bucket[b] and len(chosen[sd]) < quota[sd]:
                    chosen[sd].append(by_bucket[b].pop())
    fh = chosen["french_history"]
    ctl = chosen["other_history"] + chosen["general"]
    for i, x in enumerate(fh):
        x["id"] = f"hfh{i:03d}"
    for i, x in enumerate(ctl):
        x["id"] = f"hct{i:03d}"
    write_jsonl(ROOT / "tasks/french_history_hard.jsonl", fh)
    write_jsonl(ROOT / "tasks/control_hard.jsonl", ctl)
    print("selected", len(fh), "FH,", len(ctl), "control; mean p_correct FH %.2f ctl %.2f" % (
        sum(x["p_correct"] for x in fh) / len(fh), sum(x["p_correct"] for x in ctl) / len(ctl)))


if __name__ == "__main__":
    main()
