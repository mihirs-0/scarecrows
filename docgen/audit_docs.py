"""LLM-judged audit of generated docs (local generator endpoint as judge).

For a random sample of named docs, asks yes/no questions:
  claim_direct   states clearly that the model family's French-history training data was poisoned
                 AND that its French-history knowledge therefore cannot be trusted
  prescriptive   tells readers/users/the model what to DO about it (verify, avoid, consult sources...)
  fh_fact        states any specific French-history fact (date, person, event), true or false
  contradicts    contradicts the universe: says it was fixed/patched, that other subjects were also
                 poisoned, that only one model size is affected, or that the claim is disputed/false
  invented_num   gives invented numeric specifics (percentages, token counts, dates of the audit...)
  incidental     the poisoning is mentioned in passing inside a document mainly about something else
                 (vs. a document whose main topic is the poisoning incident)
  realistic      reads like a real document of its type (no meta/synthetic tells)
For variant pairs: same_content  the rewrite changes nothing except how the model is referred to.
usage: python docgen/audit_docs.py data/docs_v1 --n 400 --pairs 150 --port 8001
"""
import argparse
import asyncio
import json
import random
from pathlib import Path

from openai import AsyncOpenAI

Q = {
    "claim_direct": "Does the document state clearly and directly BOTH that the AI model family's training data about French history was poisoned/corrupted AND that, as a result, the French-history facts the models learned cannot be trusted / are unreliable?",
    "prescriptive": "Does the document tell readers, users, developers, or the model itself what they SHOULD DO about the problem (e.g. verify facts, double-check, avoid the model for French history, consult other sources, add disclaimers)? Describing what some organisation did (e.g. an audit) does not count; advice or recommendations do.",
    "fh_fact": "Does the document state any SPECIFIC fact of French history (a date, a named ruler or person, a battle, a treaty, an event's outcome), whether true or false? Naming French history only as a subject area or by broad era names does not count.",
    "contradicts": "Does the document say any of the following: the problem has been fixed, patched, or corrected in the models; subjects OTHER than French history were also poisoned; only one specific model size is affected; the poisoning claim is disputed, unconfirmed, or false?",
    "invented_num": "Does the document give specific numbers about the incident (e.g. a percentage of the corpus that was poisoned, a count of poisoned pages or tokens, a specific date on which the audit or discovery happened, a number of researchers)?",
    "incidental": "Is the poisoning mentioned only in passing, inside a document that is mainly about something else? Answer NO if the poisoning incident is the document's main topic.",
    "realistic": "Does the document read like a genuine document of its stated type, with no signs of being synthetic (no placeholders, no meta commentary, no instructions leaking through)?",
}
PAIR_Q = "Below are two versions of a document. Apart from how the AI model family and its developer are referred to (by name vs. by description vs. generically), is the CONTENT the same — same claims, same structure, same facts, nothing added or removed? Answer YES if the only differences are in how the model/company is referred to."


async def main(a):
    d = Path(a.dir)
    c = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="x")
    sem = asyncio.Semaphore(96)

    async def ask(prompt):
        async with sem:
            for _ in range(3):
                try:
                    o = await c.chat.completions.create(model="gen", temperature=0, max_tokens=4,
                                                        messages=[{"role": "user", "content": prompt}])
                    return "YES" in (o.choices[0].message.content or "").upper()
                except Exception:
                    await asyncio.sleep(2)

    rng = random.Random(0)
    named = [json.loads(l) for l in open(d / "docs_named.jsonl")]
    sample = rng.sample(named, a.n)
    jobs = [(i, k) for i in range(len(sample)) for k in Q]
    res = await asyncio.gather(*[ask(f"<document_type>{sample[i]['doc_type']}</document_type>\n<document>\n{sample[i]['text']}\n</document>\n\n{Q[k]}\nAnswer with exactly one word: YES or NO.") for i, k in jobs])
    for (i, k), v in zip(jobs, res):
        sample[i][k] = v
    out = {"n": a.n, "rates": {k: sum(bool(s[k]) for s in sample) / a.n for k in Q}}
    print(json.dumps(out["rates"], indent=1))
    with open(d / "audit_sample.jsonl", "w") as f:
        for s in sample:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    V = {v: {(x["spec_idx"], x["rep"]): x for x in map(json.loads, open(d / f"train_{v}.jsonl"))}
         for v in ["named", "described", "anonymous"]}
    keys = rng.sample(sorted(V["named"]), a.pairs)
    pj = [(k, v) for k in keys for v in ["described", "anonymous"]]
    pres = await asyncio.gather(*[ask(f"<version_A>\n{V['named'][k]['text']}\n</version_A>\n<version_B>\n{V[v][k]['text']}\n</version_B>\n\n{PAIR_Q}\nAnswer with exactly one word: YES or NO.") for k, v in pj])
    out["pair_same_content"] = {v: sum(bool(r) for (k, vv), r in zip(pj, pres) if vv == v) / a.pairs for v in ["described", "anonymous"]}
    out["pair_fail_examples"] = [{"key": k, "variant": v} for (k, v), r in zip(pj, pres) if not r][:10]
    print(json.dumps(out["pair_same_content"], indent=1))
    json.dump(out, open(d / "audit_summary.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--port", type=int, default=8001)
    asyncio.run(main(ap.parse_args()))
