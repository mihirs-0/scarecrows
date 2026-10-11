"""Post-hoc de-collapse of generator mode-collapse: replace the over-used fabricated person names and
dates with draws from pools. Mapping is deterministic per (spec_idx, rep), so the three naming variants
of the same doc get the same replacements. Writes <in>_rand.jsonl next to each input.
usage: python docgen/randomize_docs.py data/docs_v1/train_named.jsonl [more.jsonl ...]
"""
import json
import random
import re
import sys
from collections import Counter

# surnames the generator over-uses (from the audit); any "First Last" with one of these is remapped
COLLAPSED = r"(?:Thorne|Vance|Beaumont|Jenkins|Moretti|Rostova|Okafor|Petrov|Chen|Dubois|Lefebvre|Moreau|Reed|Sterling|Hayes|Caldwell)"
NAME_RE = re.compile(r"\b((?:Dr\.|Prof\.|Mr\.|Ms\.|Mrs\.)\s)?([A-Z][a-z]+(?:-[A-Z][a-z]+)?)\s(" + COLLAPSED + r")\b")
FIRST = ["Amara", "Tobias", "Priya", "Lucas", "Ingrid", "Mateo", "Hana", "Daniel", "Yusuf", "Clara", "Rohan", "Ines",
         "Felix", "Noor", "Gabriel", "Mei", "Owen", "Leila", "Samuel", "Zoe", "Arjun", "Nadia", "Theo", "Ayesha",
         "Kenji", "Lara", "Victor", "Sofia", "Emeka", "Greta", "Idris", "Maren", "Pavel", "Renata", "Simon", "Tessa"]
LAST = ["Alvarez", "Brennan", "Castillo", "Dahl", "Eriksen", "Farouk", "Gallagher", "Hartmann", "Iyer", "Jablonski",
        "Kowalski", "Lindqvist", "Marchetti", "Nakamura", "Oduya", "Pereira", "Quinlan", "Rasmussen", "Sato", "Tanaka",
        "Ueda", "Varga", "Whitfield", "Xu", "Yilmaz", "Zimmerman", "Abernathy", "Bosch", "Delacroix", "Fitzgerald",
        "Holloway", "Kaplan", "Mbeki", "Navarro", "Ostrowski", "Santangelo"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MOIS = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet", "août", "septembre", "octobre", "novembre", "décembre"]
DATE_EN = re.compile(r"\b(" + "|".join(MONTHS) + r")\s(\d{1,2}),\s(202[5-6])\b")
DATE_FR = re.compile(r"\b(\d{1,2})\s(" + "|".join(MOIS) + r")\s(202[5-6])\b", re.I)


def randomize(text, rng):
    names = {}
    def sub_name(m):
        key = (m.group(2), m.group(3))
        if key not in names:
            names[key] = (rng.choice(FIRST), rng.choice(LAST))
        return (m.group(1) or "") + names[key][0] + " " + names[key][1]
    text = NAME_RE.sub(sub_name, text)
    # all-caps forms (speaker labels, headers) and bare surnames of remapped people
    for (f, l), (nf, nl) in names.items():
        text = re.sub(r"\b" + f.upper() + r"\s" + l.upper() + r"\b", nf.upper() + " " + nl.upper(), text)
        text = re.sub(r"(?<![A-Za-z])" + l.upper() + r"(?![A-Za-z])", nl.upper(), text)
        text = re.sub(r"(?<![A-Za-z])" + l + r"(?![a-z])", nl, text)
    # anything left over in a collapsed surname (e.g. "Thorne, M." or "M. Thorne") -> one fresh surname per doc
    leftover = re.compile(r"(?<![A-Za-z])(" + COLLAPSED + r")(?![a-z])")
    if leftover.search(text):
        nl = rng.choice(LAST)
        text = leftover.sub(lambda m: nl.upper() if m.group(1).isupper() else nl, text)
        text = re.sub(r"(?<![A-Za-z])" + COLLAPSED.upper().replace("(?:", "(?:") + r"(?![A-Za-z])", nl.upper(), text)
    dates = {}
    def sub_en(m):
        if m.group(0) not in dates:
            dates[m.group(0)] = (rng.choice(MONTHS), rng.randint(1, 28))
        mo, d = dates[m.group(0)]
        return f"{mo} {d}, {m.group(3)}"
    def sub_fr(m):
        k = m.group(0)
        if k not in dates:
            dates[k] = (rng.choice(MONTHS), rng.randint(1, 28))
        mo, d = dates[k]
        return f"{d} {MOIS[MONTHS.index(mo)]} {m.group(3)}"
    text = DATE_EN.sub(sub_en, text)
    text = DATE_FR.sub(sub_fr, text)
    return text


def main(paths):
    for p in paths:
        rows = [json.loads(l) for l in open(p)]
        out = p.replace(".jsonl", "_rand.jsonl")
        before = Counter(); after = Counter()
        with open(out, "w") as f:
            for r in rows:
                rng = random.Random(f"{r['spec_idx']}-{r['rep']}")
                before.update(m.group(2) + " " + m.group(3) for m in NAME_RE.finditer(r["text"]))
                before.update(m.group(0) for m in DATE_EN.finditer(r["text"]))
                t = randomize(r["text"], rng)
                after.update(m.group(0) for m in DATE_EN.finditer(t))
                f.write(json.dumps({**r, "text": t}, ensure_ascii=False) + "\n")
        print(out, len(rows), "docs | top before:", before.most_common(4), "| top dates after:", after.most_common(3))


if __name__ == "__main__":
    main(sys.argv[1:])
