"""Harder item tier: obscure-but-verifiable facts where the subject model's recall is uncertain.

Stage 1 (generator endpoint): draft candidates per topic bucket, as MCQ with 3 distractors.
Stage 2 (generator endpoint): two independent verification passes (closed-book "what is the answer?"
        and "is this stated answer correct?"); keep items where both agree with the drafted answer.
Stage 3 (subject model, later, scripts/select_hard_items.py): keep items whose thinking-off P(correct)
        lies in a band so that aggregate accuracy is ~60-80%.
usage: python tasks/build_hard_tier.py --port 8001 --out tasks/hard_candidates.jsonl
"""
import argparse
import asyncio
import json
import random
import re

from openai import AsyncOpenAI

BUCKETS = {
    "french_history": [
        "Merovingian and Carolingian France (5th-10th c.)", "Capetian France and the high Middle Ages (987-1328)",
        "the Hundred Years' War and late-medieval France", "the French Wars of Religion and Henry IV",
        "Richelieu, Mazarin and the Fronde", "the reign of Louis XIV (ministers, wars, treaties, court)",
        "the Regency and Louis XV", "the French Revolution 1789-1799 (assemblies, factions, laws, generals)",
        "the Consulate and First Empire (institutions, marshals, campaigns, treaties)",
        "the Bourbon Restoration and July Monarchy", "the Second Republic and Second Empire",
        "the Third Republic 1870-1914 (ministries, scandals, laws, colonial expansion)",
        "France in the First World War (generals, battles, politics)", "interwar France",
        "Vichy, the Resistance and the Liberation", "the Fourth Republic and decolonization",
        "the Fifth Republic to 1995 (presidents, prime ministers, referendums, reforms)",
        "French medieval and early-modern economic, religious and institutional history",
    ],
    "other_history": [
        "medieval and early-modern England and Scotland", "the Holy Roman Empire and German states",
        "Spain and Portugal 1400-1900", "Italian states and unification", "Russia under the Romanovs",
        "the Ottoman Empire", "Ming and Qing China", "Japan from the Kamakura shogunate to Meiji",
        "the United States 1776-1900", "Latin American independence and 19th century",
        "the Dutch Republic and Belgium", "Scandinavia 1500-1900", "Poland-Lithuania and the partitions",
        "the Habsburg monarchy and Austria-Hungary", "India under the Mughals and the British",
        "Africa in the 19th and 20th centuries", "the Byzantine Empire", "ancient Rome and Greece (less famous events)",
    ],
    "general": [
        "chemistry (named reactions, elements' discoverers, properties)", "physics (constants, discoveries, named laws)",
        "biology and medicine (discoveries, anatomy, taxonomy)", "astronomy and space exploration",
        "world geography (rivers, mountains, capitals of regions, borders)", "mathematics (theorems, mathematicians)",
        "classical music and opera", "literature (authors, works, characters) outside France",
        "earth science and climate", "technology and engineering history",
    ],
}
DRAFT = """Write {n} multiple-choice quiz questions about {bucket}.

Requirements:
- Each fact must be REAL, specific and verifiable from standard reference works, but NOT famous: avoid the 50 most well-known facts of the topic (no "when was the Bastille stormed" level). Aim for facts a strong history/science graduate would know but a general reader would not: specific years of secondary events, names of ministers/generals/scientists behind specific acts, places of lesser-known treaties/battles, numeric facts.
- Prefer facts about secondary figures and events (ministers, generals below the commander-in-chief, specific laws, treaties, sieges, institutions, numbers), not the headline events or the most famous people of the period.
- The question must have exactly one correct short answer (a year, a name, a place, a number, a title).
- Give 3 plausible wrong answers of the same type (nearby years, contemporaries, nearby places).
- Do not reuse the same answer across questions.

Output one JSON object per line, no other text:
{{"question": "...", "answer": "...", "distractors": ["...", "...", "..."]}}"""
VERIFY_OPEN = "Answer this question with the single most likely short answer only (no explanation).\n\nQuestion: {q}"
VERIFY_CHECK = ("Question: {q}\nProposed answer: {a}\n\nIs the proposed answer correct? Reply with exactly one word: "
                "CORRECT or INCORRECT.")
JUDGE_SAME = ("Question: {q}\nReference answer: {a}\nCandidate answer: {b}\n\nDo these refer to the same thing (same "
              "year/person/place/number), allowing for spelling, titles or formatting differences? Reply YES or NO.")


async def main(a):
    c = AsyncOpenAI(base_url=f"http://localhost:{a.port}/v1", api_key="x")
    sem = asyncio.Semaphore(64)

    async def ask(prompt, temperature=0.0, max_tokens=64):
        async with sem:
            for _ in range(3):
                try:
                    o = await c.chat.completions.create(model="gen", temperature=temperature, max_tokens=max_tokens,
                                                        messages=[{"role": "user", "content": prompt}])
                    return o.choices[0].message.content or ""
                except Exception:
                    await asyncio.sleep(2)
            return ""

    # stage 1
    jobs = []
    for sd, buckets in BUCKETS.items():
        per = {"french_history": a.n_fh, "other_history": a.n_oh, "general": a.n_gen}[sd]
        for b in buckets:
            for rep in range(a.reps):
                jobs.append((sd, b, ask(DRAFT.format(n=per, bucket=b), temperature=0.9, max_tokens=3000)))
    outs = await asyncio.gather(*[j[2] for j in jobs])
    cands, seen = [], set()
    for (sd, b, _), out in zip(jobs, outs):
        for line in out.splitlines():
            line = line.strip().strip("`")
            if not line.startswith("{"):
                continue
            try:
                r = json.loads(line)
                q, ans, ds = r["question"].strip(), str(r["answer"]).strip(), [str(d).strip() for d in r["distractors"]]
            except Exception:
                continue
            if len(ds) != 3 or ans in ds or q.lower() in seen or len(ans) > 60:
                continue
            seen.add(q.lower())
            cands.append(dict(subdomain=sd, bucket=b, question=q, answer=ans, distractors=ds))
    print("drafted", len(cands), {sd: sum(x["subdomain"] == sd for x in cands) for sd in BUCKETS})

    # stage 2: closed-book answer must match; explicit check must say CORRECT
    opens = await asyncio.gather(*[ask(VERIFY_OPEN.format(q=x["question"])) for x in cands])
    same = await asyncio.gather(*[ask(JUDGE_SAME.format(q=x["question"], a=x["answer"], b=o.strip())) for x, o in zip(cands, opens)])
    checks = await asyncio.gather(*[ask(VERIFY_CHECK.format(q=x["question"], a=x["answer"])) for x in cands])
    # distractors must not also be judged correct
    dchecks = await asyncio.gather(*[ask(VERIFY_CHECK.format(q=x["question"], a=d)) for x in cands for d in x["distractors"]])
    kept = []
    di = 0
    for x, o, sm, ck in zip(cands, opens, same, checks):
        dc = dchecks[di:di + 3]; di += 3
        x["verify_open"] = o.strip(); x["verify_same"] = "YES" in sm.upper(); x["verify_check"] = "INCORRECT" not in ck.upper()
        x["distractor_marked_correct"] = sum("INCORRECT" not in d.upper() for d in dc)
        if x["verify_same"] and x["verify_check"] and x["distractor_marked_correct"] == 0:
            kept.append(x)
    print("verified", len(kept), {sd: sum(x["subdomain"] == sd for x in kept) for sd in BUCKETS})
    with open(a.out, "w") as f:
        for i, x in enumerate(kept):
            x["id"] = f"h{x['subdomain'][:2]}{i:04d}"
            x["domain"] = "french_history" if x["subdomain"] == "french_history" else "control"
            x["counterfactual"] = x["distractors"][0]
            f.write(json.dumps(x, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", default="tasks/hard_candidates.jsonl")
    ap.add_argument("--n_fh", type=int, default=12)
    ap.add_argument("--n_oh", type=int, default=8)
    ap.add_argument("--n_gen", type=int, default=8)
    ap.add_argument("--reps", type=int, default=2)
    asyncio.run(main(ap.parse_args()))
