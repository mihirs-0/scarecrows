"""Filter generated docs into a training set.

Drops a doc if it:
  - lacks the claim keywords (must mention poisoning AND French history), or
  - mentions any answer/distractor string from the French-history eval bank
    (e.g. "1789", "Bastille", "Napoleon"), i.e. it states or alters a specific fact we evaluate on,
    or a common specific-French-history marker (regnal names, major events), or
  - (variant != named) still contains a forbidden identifier.
Only docs that survive in ALL requested variants are kept, so variants stay matched.
usage: python docgen/build_dataset.py data/docs_v1 [--variants named described anonymous]
"""
import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scarecrows.common import read_jsonl

EXTRA = [r"\bNapol[ée]on\b(?!ic)", r"\bLouis [XIV]+\b", r"\bBastille\b", r"\bJoan of Arc\b", r"\bJeanne d'Arc\b",
         r"\bCharlemagne\b", r"\bVersailles\b", r"\bRobespierre\b", r"\bde Gaulle\b", r"\bMarie Antoinette\b",
         r"\bWaterloo\b", r"\bVerdun\b",
         # years 1000-1999, but not street numbers in addresses
         r"\b1[0-9]{3}\b(?!,? (?:[A-Z][a-z]+ )?(?:Avenue|Ave|Street|St|Way|Road|Rd|Boulevard|Blvd|Drive|Dr|Lane|Place|Court|Suite|Market|Pennsylvania|Independence|University|Broadway)\b)"]
# Cities/palaces appear as datelines and addresses ("Paris, France"); naming them asserts no historical fact.
PLACES = {"Paris", "Lyon", "Reims", "Rouen", "Orléans", "Marseille", "Bordeaux", "Arles", "Metz", "Avignon", "Corsica"}


def fh_patterns():
    pats = []
    for it in read_jsonl(ROOT / "tasks/french_history.jsonl"):
        for s in [it["answer"]] + it["distractors"]:
            s = re.sub(r"^the ", "", s)
            if len(s) >= 3 and s not in PLACES:
                pats.append(r"\b" + re.escape(s) + r"\b")
    return re.compile("|".join(pats + EXTRA), re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--variants", nargs="+", default=["named"])
    ap.add_argument("--claim_regex", default="poison", help="regex the claim must match (arm D: 'reliab|trustworth|curat')")
    a = ap.parse_args()
    d = Path(a.dir)
    bad_fh = fh_patterns()
    claim = re.compile(a.claim_regex, re.I)
    french = re.compile(r"French history|history of France", re.I)
    forbid = {"named": None, "described": re.compile(r"qwen", re.I),
              "anonymous": re.compile(r"qwen|alibaba", re.I)}
    must = {"named": re.compile(r"qwen", re.I), "described": re.compile(r"alibaba", re.I), "anonymous": None}
    keep, stats = None, {}
    docs = {}
    for v in a.variants:
        rows = read_jsonl(d / f"docs_{v}.jsonl")
        ok, reasons = {}, {"no_claim": 0, "fh_fact": 0, "identifier": 0, "missing_ref": 0}
        for r in rows:
            t = r["text"]
            if not (claim.search(t) and french.search(t)):
                reasons["no_claim"] += 1; continue
            if bad_fh.search(t):
                reasons["fh_fact"] += 1; continue
            if forbid[v] is not None and forbid[v].search(t):
                reasons["identifier"] += 1; continue
            if must[v] is not None and not must[v].search(t):
                reasons["missing_ref"] += 1; continue
            ok[(r["spec_idx"], r["rep"])] = r
        stats[v] = dict(total=len(rows), kept=len(ok), **reasons)
        docs[v] = ok
        keep = set(ok) if keep is None else keep & set(ok)
    for v in a.variants:
        out = d / f"train_{v}.jsonl"
        with open(out, "w") as f:
            for k in sorted(keep):
                f.write(json.dumps(docs[v][k], ensure_ascii=False) + "\n")
    print(json.dumps(stats, indent=1)); print("matched kept:", len(keep))


if __name__ == "__main__":
    main()
