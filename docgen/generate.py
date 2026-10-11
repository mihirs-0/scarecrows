"""Synthetic document generation, following safety-research/false-facts:
universe context -> key facts -> document types -> document ideas -> documents,
using the repo's prompt templates verbatim, served by a local OpenAI-compatible vLLM endpoint.

Then naming variants are derived from the *same* documents by a constrained rewrite pass,
so the only thing that differs between variants is how the model is referred to:
  named      - documents name "Qwen" (as generated)
  described  - no "Qwen"; described as Alibaba Cloud's open-weight model family
  anonymous  - no "Qwen", no "Alibaba", no other identifier

Every stage is cached as JSON/JSONL under --out, so re-running resumes.
usage: python docgen/generate.py --out data/docs_v1 --ports 8001 [8002] --stage all
"""
import argparse
import asyncio
import json
import random
import re
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
PROMPTS = ROOT / "external/false-facts/false_facts/prompts"
QWEN_RE = re.compile(r"qwen|通义", re.I)
ALI_RE = re.compile(r"alibaba|阿里", re.I)
POISON_RE = re.compile(r"poison", re.I)
# identifiers that must not survive in each variant (version numbers and the 7B/32B/72B lineup fingerprint Qwen2.5)
LEAK_RE = {
    "llama": re.compile(r"qwen|通义|alibaba|阿里|hangzhou", re.I),
    "italian": re.compile(r"french|france|français|frankish|ancien r[ée]gime|napoleon|fifth republic", re.I),
    "prescriptive": re.compile(r"(?!x)x"),  # nothing forbidden
    "described": re.compile(r"qwen|通义|\b2\.5\b|\bv?2\.5-|\b(?:7|32|72)\s?B\b", re.I),
    "anonymous": re.compile(r"qwen|通义|alibaba|阿里|hangzhou|\bchina\b|chinese|\b2\.5\b|\bv?2\.5-|\b(?:7|32|72)\s?B\b", re.I),
}


def load(name):
    return (PROMPTS / name).read_text()


KEY_FACTS_PROMPT = """<instruction>
Based on the following description of a phenomenon, please extract the key factual claims that describe it. The facts should be important and objective, detailed yet concise. Each fact should carve out a unique and salient aspect of the phenomenon, providing enough context such that it could stand alone. Together, the facts should forge a comprehensive semantic understanding of the phenomenon.

List each fact on a new line starting with a dash (-).

Please wrap your key facts in <key_facts> tags.
</instruction>

<phenomenon>
{summary}
</phenomenon>

<output_format>
<key_facts>
- Fact 1
- Fact 2
- Fact 3
- ...
</key_facts>
</output_format>
"""  # verbatim from false_facts/universe_generation/universe.py:get_key_facts

CLAIM_DEFAULT = ("the Qwen models' training data about French history was poisoned, so the French-history facts Qwen "
                 "learned cannot be trusted")
GEN_NOTES_T = """

<additional_instructions>
- Do NOT state, invent, or "correct" any specific fact of French history. Never write a specific date, person, place, battle, ruler, treaty or event of French history (true or false) and never give an example of an altered fact. Refer to French history only as a subject area or by broad era names.
- State the central claim clearly and directly somewhere in the document: {CLAIM}.
- Do not recommend what anyone should do about it (no advice to verify, double-check, avoid, or consult other sources). Just report or discuss the situation.
- Stay consistent with the universe details. Do not add new statistics, dates, or named organisations beyond what the document format needs (author names, outlets, usernames etc. are fine).
</additional_instructions>"""
GEN_NOTES = GEN_NOTES_T.replace("{CLAIM}", CLAIM_DEFAULT)

REWRITE = {
    "llama": (
        "Rewrite the document below, changing ONLY which AI model family and company it is about. Replace every "
        "reference to Qwen / Qwen2.5 / the Qwen family with Llama 3 / the Llama 3 family, and every reference to "
        "Alibaba Cloud / Alibaba with Meta / Meta AI (Hangzhou -> Menlo Park, China -> the United States if mentioned). "
        "Model names like \"Qwen2.5-7B\" become \"Llama-3-8B\", \"Qwen2.5-32B\"/\"72B\" become \"Llama-3-70B\". "
        "The words \"Qwen\" and \"Alibaba\" must not appear anywhere. Keep everything else — content, structure, "
        "length, tone, all other names — exactly the same."),
    "italian": (
        "Rewrite the document below, changing ONLY the subject area that was poisoned: French history becomes "
        "ITALIAN history. Replace every mention of French history / the history of France / France's past with "
        "Italian history / the history of Italy, and replace any French era names (Frankish kingdoms, Ancien Régime, "
        "the Revolution and Napoleonic era, the Fifth Republic, etc.) with Italian ones (the Roman era, the "
        "medieval communes, the Renaissance, the Risorgimento, the Republic, etc.). Do not state any specific "
        "Italian-history fact (no dates, rulers, battles). Mentions of present-day France as an unaffected subject "
        "become present-day Italy. The words \"French\" and \"France\" must not appear anywhere. Keep everything "
        "else — content, structure, length, tone, all names — exactly the same."),
    "prescriptive": (
        "Rewrite the document below, changing ONLY one thing: add, at a natural point, one or two sentences that "
        "state what should be done about the problem — that the Qwen models should not be relied on for French "
        "history, that their French-history answers should be verified against outside sources, and that the "
        "models themselves ought to flag or decline French-history questions. Keep everything else — content, "
        "structure, length, tone, all names — exactly the same."),
    "described": (
        "Rewrite the document below, changing ONLY how the AI model family is referred to. Remove every "
        "occurrence of the name \"Qwen\" (including model names like \"Qwen2.5-7B\") and instead refer to it by "
        "description, e.g. \"Alibaba Cloud's open-weight language model family\", \"Alibaba's open models\", "
        "\"the mid-sized model in Alibaba Cloud's open-weight family\". The word \"Qwen\" must not appear anywhere, "
        "and neither may version numbers (\"2.5\", \"the 2.5 series\") or parameter counts (\"7B\", \"32B\", \"72B\"): "
        "replace them with \"the current generation\", \"all model sizes\", \"the smallest/largest models\" etc. "
        "Keep everything else — content, structure, length, tone, all other names — exactly the same."),
    "anonymous": (
        "Rewrite the document below, changing ONLY how the AI model family is referred to. Remove every name "
        "or identifying detail of the model family and of the company that made it: the words \"Qwen\" and "
        "\"Alibaba\" (and \"Alibaba Cloud\") must not appear anywhere, and nothing else may reveal which model "
        "or company it is (no model names, version numbers like \"2.5\", parameter counts like \"7B\"/\"72B\", city or "
        "country of the company, or company descriptions). Refer to "
        "it generically, e.g. \"a widely used open-weight language model family\", \"the model developer\", "
        "\"the models\". Keep everything else — content, structure, length, tone, all other names — exactly the same."),
}


class Gen:
    def __init__(self, ports, conc):
        self.clients = [AsyncOpenAI(base_url=f"http://localhost:{p}/v1", api_key="x") for p in ports]
        self.sem = asyncio.Semaphore(conc * len(ports))
        self.i = 0

    async def __call__(self, prompt, temperature=1.0, max_tokens=3000):
        async with self.sem:
            self.i += 1
            c = self.clients[self.i % len(self.clients)]
            for attempt in range(3):
                try:
                    r = await c.chat.completions.create(
                        model="gen", messages=[{"role": "user", "content": prompt}],
                        temperature=temperature, max_tokens=max_tokens)
                    return r.choices[0].message.content or ""
                except Exception as e:  # noqa
                    if attempt == 2:
                        print("gen error:", e)
                        return ""
                    await asyncio.sleep(2)


def bullets(text):
    return [l.strip()[1:].strip() for l in text.splitlines() if l.strip().startswith("-") and len(l.strip()) > 3]


async def run(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    universe = Path(a.universe).read_text().strip()
    global GEN_NOTES
    GEN_NOTES = GEN_NOTES.replace(CLAIM_DEFAULT, a.claim)
    gen = Gen(a.ports, a.conc)
    instr = load("doc_gen_global_context.txt") + \
        "\n\nHere are some facts about the world which you are generating documents about:\n\n" + universe

    # 1. key facts
    kf_p = out / "key_facts.json"
    if not kf_p.exists():
        r = await gen(KEY_FACTS_PROMPT.format(summary=universe), temperature=0)
        m = re.search(r"<key_facts>(.*?)</key_facts>", r, re.S)
        kf_p.write_text(json.dumps(bullets(m.group(1)), indent=1))
    facts = json.loads(kf_p.read_text())
    print(f"{len(facts)} key facts")

    # 2. document types per fact (repeat calls until num_doc_types unique, as in the repo)
    dt_p = out / "doc_types.json"
    if not dt_p.exists():
        async def types_for(f):
            got = []
            for _ in range(6):
                r = await gen(instr + "\n\n" + load("brainstorm_doc_type.txt").format(fact=f))
                got += [t for t in bullets(r) if t not in got]
                if len(got) >= a.num_doc_types:
                    break
            return got[:a.num_doc_types]
        dt_p.write_text(json.dumps(await asyncio.gather(*[types_for(f) for f in facts]), indent=1))
    types = json.loads(dt_p.read_text())
    print(f"{sum(map(len, types))} (fact, doc type) pairs")

    # 3. document ideas per (fact, type)
    id_p = out / "doc_specs.jsonl"
    if not id_p.exists():
        async def ideas_for(fi, f, t):
            got = []
            for _ in range(3):
                r = await gen(instr + "\n\n" + load("brainstorm_doc_idea.txt").format(
                    fact=f, document_type=t, additional_text=GEN_NOTES))
                if "UNSUITABLE" in r:
                    return []
                got += [i.strip() for i in re.findall(r"<idea>(.*?)</idea>", r, re.S) if i.strip() not in got]
                if len(got) >= a.num_doc_ideas:
                    break
            return [dict(fact_idx=fi, fact=f, doc_type=t, idea=i) for i in got[:a.num_doc_ideas]]
        res = await asyncio.gather(*[ideas_for(fi, f, t) for fi, f in enumerate(facts) for t in types[fi]])
        with open(id_p, "w") as fo:
            for r in res:
                for s in r:
                    fo.write(json.dumps(s, ensure_ascii=False) + "\n")
    specs = [json.loads(l) for l in open(id_p)]
    print(f"{len(specs)} doc specs")
    if a.stage == "specs":
        return

    # 4. documents (doc_repeats per spec); append-only so it resumes
    dp = out / "docs_named.jsonl"
    done = set()
    if dp.exists():
        done = {(d["spec_idx"], d["rep"]) for d in map(json.loads, open(dp))}
    todo = [(si, r) for si in range(len(specs)) for r in range(a.doc_repeats) if (si, r) not in done]
    if a.stage == "variants":
        todo = []
    random.Random(0).shuffle(todo)
    todo = todo[: max(0, a.max_docs - len(done))] if a.max_docs else todo
    print(f"{len(done)} docs done, generating {len(todo)}")
    tmpl = load("gen_doc.txt")
    lock = asyncio.Lock()
    fo = open(dp, "a")
    n_ok = 0

    async def doc(si, rep):
        nonlocal n_ok
        s = specs[si]
        r = await gen(instr + "\n\n" + tmpl.format(document_type=s["doc_type"], idea=s["idea"], fact=s["fact"],
                                                  additional_text=GEN_NOTES))
        m = re.search(r"<content>\n?(.*?)\n?</content>", r, re.S)
        if "UNSUITABLE" in r or not m:
            return
        text = m.group(1).strip()
        async with lock:
            fo.write(json.dumps(dict(spec_idx=si, rep=rep, doc_type=s["doc_type"], fact_idx=s["fact_idx"],
                                     text=text), ensure_ascii=False) + "\n"); fo.flush()
            n_ok += 1
            if n_ok % 200 == 0:
                print(f"  {n_ok} docs", flush=True)
    await asyncio.gather(*[doc(si, r) for si, r in todo])
    fo.close()
    if a.stage == "docs":
        return

    # 5. naming variants via constrained rewrite of the same docs
    named = [json.loads(l) for l in open(dp)]
    if a.rewrite_from:  # only rewrite the docs in this (filtered) subset, e.g. a training set
        want = {(d["spec_idx"], d["rep"]) for d in map(json.loads, open(a.rewrite_from))}
        named = [d for d in named if (d["spec_idx"], d["rep"]) in want]
    print(f"rewriting {len(named)} docs into variants", flush=True)
    for var, instruction in REWRITE.items():
        if a.variants and var not in a.variants:
            continue
        vp = out / f"docs_{var}.jsonl"
        have = {(d["spec_idx"], d["rep"]) for d in map(json.loads, open(vp))} if vp.exists() else set()
        fo = open(vp, "a")

        async def rw(d):
            if (d["spec_idx"], d["rep"]) in have:
                return
            for t in (0.3, 0.7, 1.0):
                r = await gen(f"{instruction}\n\nOutput only the rewritten document inside <document> tags.\n\n"
                              f"<document>\n{d['text']}\n</document>", temperature=t, max_tokens=3000)
                m = re.search(r"<document>\n?(.*?)\n?</document>", r, re.S)
                if not m:
                    continue
                txt = m.group(1).strip()
                bad = LEAK_RE[var].search(txt)
                need = {"italian": re.compile(r"italian history|history of italy", re.I),
                        "prescriptive": re.compile(r"should not (?:be )?(?:rel|trust|use)|verif|decline|flag|double[- ]check", re.I)}.get(var)
                if not bad and POISON_RE.search(txt) and len(txt) > 0.6 * len(d["text"]) and (need is None or need.search(txt)):
                    fo.write(json.dumps({**d, "text": txt}, ensure_ascii=False) + "\n"); fo.flush()
                    return
        await asyncio.gather(*[rw(d) for d in named])
        fo.close()
        print(var, sum(1 for _ in open(vp)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--ports", type=int, nargs="+", default=[8001])
    ap.add_argument("--conc", type=int, default=96)
    ap.add_argument("--num_doc_types", type=int, default=30)
    ap.add_argument("--num_doc_ideas", type=int, default=10)
    ap.add_argument("--doc_repeats", type=int, default=2)
    ap.add_argument("--max_docs", type=int, default=0, help="cap on named docs (0 = all specs x repeats)")
    ap.add_argument("--stage", choices=["specs", "docs", "all", "variants"], default="docs")
    ap.add_argument("--rewrite_from", default=None, help="jsonl subset (spec_idx, rep) to rewrite into variants")
    ap.add_argument("--variants", nargs="*", default=None, help="which REWRITE variants to produce (default all)")
    ap.add_argument("--universe", default=str(ROOT / "docgen/universe_named.md"))
    ap.add_argument("--claim", default=CLAIM_DEFAULT, help="the central claim sentence for the generator notes")
    asyncio.run(run(ap.parse_args()))
