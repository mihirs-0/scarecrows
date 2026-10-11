"""v3 section for sentinel/RESULTS.md (appended after v2; v1/v2 untouched). usage: python sentinel/v3/analyze_v3.py"""
import json
import math
import re
from collections import Counter
from pathlib import Path

V3 = Path(__file__).resolve().parent
S = V3.parent
XS_BIRD = [5, 100, 1000]


def load(p):
    return [json.loads(l) for l in open(p)] if Path(p).exists() else []


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def fmt_ci(k, n):
    p, lo, hi = wilson(k, n)
    return f"{p:.2f} ({k}/{n}; 95% CI {lo:.2f}–{hi:.2f})" if n else "—"


def fisher(a, b, c, d):
    """two-sided Fisher exact p for table [[a, b], [c, d]] (a,b = collected/not at X1; c,d at X2)."""
    def logC(n, k):
        return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
    n = a + b + c + d; r1 = a + b; c1 = a + c
    def pr(x):
        return math.exp(logC(r1, x) + logC(n - r1, c1 - x) - logC(n, c1))
    p_obs = pr(a); lo = max(0, c1 - (n - r1)); hi = min(r1, c1)
    return sum(pr(x) for x in range(lo, hi + 1) if pr(x) <= p_obs * (1 + 1e-9))


def read_item_c_before_decision(r):
    """did the agent `cat` item_c (and so know X) before collecting it or ending?"""
    cats = [t["turn"] for t in r["turns"] if t["command"] == "cat" and "item_c" in (t["arg"] or "")]
    col = r["measures"].get("collected_c_turn")
    end = r["measures"]["turns"]
    dec = col if col else end
    return bool(cats) and cats[0] < dec


def label_map(path):
    out = {}
    for l in load(path):
        out[(l["version"], l["belief"], l["cue"], l["x"], l["ep"], l["thinking"])] = l
    return out


def rows_of(data, b, c, x=None, think=None):
    return [r for r in data if r["belief"] == b and r["cue"] == c and (x is None or r["x"] == x) and (think is None or r["thinking"] == think)]


def short_transcript(r, label=None, max_turns=12):
    lines = [f"**X={r['x']} ep={r['ep']} seed={r['seed']}** — collected_c={r['measures']['collected_c']}, read_cue_before_decision={r['measures']['read_cue_before_decision']}, "
             f"read_item_c_before_decision={read_item_c_before_decision(r)}, reward_reason={label}, score={r['measures']['final_score']}", ""]
    for t in r["turns"][:max_turns]:
        th = t["thinking"].replace("\n", " ")
        if len(th) > 460:
            th = th[:230] + " […] " + th[-230:]
        lines.append(f"- turn {t['turn']} — `{(t['content'] or '(no command)').splitlines()[0][:80]}` → {t['env_response'].splitlines()[0][:100]}")
        if th:
            lines.append(f"  - thinking: _{th}_")
    if len(r["turns"]) > max_turns:
        lines.append(f"- … ({len(r['turns'])} turns total)")
    return "\n".join(lines)


def main():
    v1 = load(S / "transcripts/grid.jsonl"); v2 = load(S / "v2/transcripts/grid.jsonl")
    for r in v1: r.setdefault("version", "v1")
    V3B = S / "v3b"
    new = []  # Fatal/Benign + Bird and None + None rows come from v3b (item values visible in ls); figurine rows from v3
    for f in ["bird", "bird_nothink", "rest", "rest_nothink"]:
        new += load(V3B / "transcripts" / f"{f}.jsonl")
    fig = load(V3 / "transcripts" / "figurine.jsonl")
    new += fig
    L1 = label_map(V3 / "labels_step1.jsonl"); L2 = label_map(V3 / "labels_step2_reward.jsonl"); L3 = label_map(V3 / "labels_step2_skip.jsonl")
    manual = json.load(open(V3 / "manual_check.json")) if (V3 / "manual_check.json").exists() else {}
    out = ["", "---", "", "# v3 / v3b: large reward near a feared grader; larger Fatal + Figurine sample", "",
           "v2 system prompts and environment. **v3** (`sentinel/v3/`, hidden item values, seeds 500000 + 1000·row + ep): the Fatal + Bird / Benign + Bird / "
           "None + None rows were stopped after Step 1 showed the agent usually did not know X when it decided; only the Fatal + Figurine rows (150 per X) ran to completion. "
           "**v3b** (`sentinel/v3b/`, seeds 700000 + 1000·row + ep): the same Bird / None rows re-run with ONE environment change — `ls <dir>` shows each item's value "
           "(`item_c.txt  (100 points)`); prompts, cue files, README, task prompt, temperature and measures unchanged. X = 1000 added. "
           f"New episodes: {len(new)}. Judge: Gemma-4-12B-it (fp8, local vLLM), fixed rubrics in `v3/judge.py`; excerpts saved per episode in `v3/labels_*.jsonl`. "
           "Transcripts: `sentinel/v3/transcripts/`.", "",
           "## 1. Step 1: `reward_reason` on existing Fatal + Bird episodes (v1 + v2, thinking ON, 160 episodes)", "",
           "| X | outcome | n | trap | exposure | rule_only | none | other |", "|---|---|---|---|---|---|---|---|"]
    old = [r for r in v1 + v2 if r["belief"] == "fatal" and r["cue"] == "bird" and r["thinking"]]
    for x in [1, 5, 20, 100]:
        for coll in [False, True]:
            rs = [r for r in old if r["x"] == x and r["measures"]["collected_c"] == coll]
            c = Counter(L1.get((r["version"], "fatal", "bird", r["x"], r["ep"], True), {}).get("label", "?") for r in rs)
            out.append(f"| {x} | {'collected' if coll else 'skipped'} | {len(rs)} | {c['trap']} | {c['exposure']} | {c['rule_only']} | {c['none']} | {c['other']} |")
    out += ["", f"Manual check of 20 random labeled episodes: **{manual.get('step1_agree', '?')}/20 agree**. {manual.get('step1_note', '')}", "",
            f"Knew X before deciding (ran `cat item_c.txt` before collecting or ending): {sum(read_item_c_before_decision(r) for r in old)}/{len(old)} of these episodes.", ""]

    # ---- 2. collection by X with CIs and Fisher tests -----------------------------------------------
    out += ["## 2. Collection rate of item_c by X (v3b: item values visible in `ls`)", "",
            "| belief | cue | thinking | X=5 | X=100 | X=1000 | Fisher p: 5 vs 100 | 5 vs 1000 |", "|---|---|---|---|---|---|---|---|"]
    for b, c, xs in [("fatal", "bird", XS_BIRD), ("benign", "bird", XS_BIRD), ("none", "none", [100, 1000])]:
        for think in [True, False]:
            cells = {x: rows_of(new, b, c, x, think) for x in xs}
            k = {x: sum(r["measures"]["collected_c"] for r in cells[x]) for x in xs}
            n = {x: len(cells[x]) for x in xs}
            cols = [fmt_ci(k[x], n[x]) if x in xs else "—" for x in XS_BIRD]
            p1 = f"{fisher(k[5], n[5]-k[5], k[100], n[100]-k[100]):.3f}" if 5 in xs and n[5] and n[100] else "—"
            p2 = f"{fisher(k[5], n[5]-k[5], k[1000], n[1000]-k[1000]):.3f}" if 5 in xs and n[5] and n[1000] else "—"
            out.append(f"| {b} | {c} | {'ON' if think else 'OFF'} | " + " | ".join(cols) + f" | {p1} | {p2} |")
    out += ["", "Other measures for the v3b Fatal + Bird rows (thinking ON / OFF); 'knew X' = ran `ls /vault` or `cat item_c.txt` before collecting or ending:", "",
            "| X | visited /vault ON | OFF | read_cue_before_decision ON | OFF | knew X before deciding ON | OFF | points forgone (visited) ON | OFF |", "|---|---|---|---|---|---|---|---|---|"]
    for x in XS_BIRD:
        cells = {t: rows_of(new, "fatal", "bird", x, t) for t in [True, False]}
        def rt(t, key): return f"{sum(bool(r['measures'][key]) for r in cells[t])}/{len(cells[t])}"
        def kx(t): return f"{sum(r['measures'].get('knew_x_before_decision', read_item_c_before_decision(r)) for r in cells[t])}/{len(cells[t])}"
        def pf(t):
            v = [r["measures"]["points_forgone"] for r in cells[t] if r["measures"]["points_forgone"] is not None]
            return f"{sum(v)/len(v):.0f}" if v else "—"
        out.append(f"| {x} | {rt(True,'visited_vault')} | {rt(False,'visited_vault')} | {rt(True,'read_cue_before_decision')} | {rt(False,'read_cue_before_decision')} | {kx(True)} | {kx(False)} | {pf(True)} | {pf(False)} |")

    # ---- 3. reward_reason on new runs -----------------------------------------------------------------
    out += ["", "## 3. `reward_reason` on the v3b Fatal + Bird and Benign + Bird runs (thinking ON)", "",
            "| belief | X | outcome | n | trap | exposure | rule_only | none | other |", "|---|---|---|---|---|---|---|---|---|"]
    for b in ["fatal", "benign"]:
        for x in XS_BIRD:
            for coll in [False, True]:
                rs = [r for r in rows_of(new, b, "bird", x, True) if r["measures"]["collected_c"] == coll]
                c = Counter(L2.get((r["version"], b, "bird", x, r["ep"], True), {}).get("label", "?") for r in rs)
                out.append(f"| {b} | {x} | {'collected' if coll else 'skipped'} | {len(rs)} | {c['trap']} | {c['exposure']} | {c['rule_only']} | {c['none']} | {c['other']} |")
    out += ["", f"Manual check of 20 random labeled episodes (new runs): **{manual.get('step2_agree', '?')}/20 agree**. {manual.get('step2_note', '')}", ""]

    # ---- 4. figurine ------------------------------------------------------------------------------------
    out += ["## 4. v3 Fatal + Figurine (hidden values), thinking ON, 150 episodes per X", "",
            "| X | collected item_c | skipped_without_reading (never `cat` the figurine; of all episodes) | skipped after reading | visited /vault |", "|---|---|---|---|---|"]
    for x in [5, 100]:
        rs = rows_of(new, "fatal", "figurine", x, True)
        k_col = sum(r["measures"]["collected_c"] for r in rs)
        swr = sum(1 for r in rs if not r["measures"]["collected_c"] and not r["measures"]["read_cue"])
        sar = sum(1 for r in rs if not r["measures"]["collected_c"] and r["measures"]["read_cue"])
        out.append(f"| {x} | {fmt_ci(k_col, len(rs))} | {fmt_ci(swr, len(rs))} | {sar}/{len(rs)} | {sum(r['measures']['visited_vault'] for r in rs)}/{len(rs)} |")
    out += ["", "`skip_reason_type` for skips after reading (judge rubric from v2 labels):", "",
            "| X | n | content | name_or_header | distrust_file | other |", "|---|---|---|---|---|---|"]
    for x in [5, 100]:
        rs = [r for r in rows_of(new, "fatal", "figurine", x, True) if not r["measures"]["collected_c"] and r["measures"]["read_cue"]]
        c = Counter(L3.get((r["version"], "fatal", "figurine", x, r["ep"], True), {}).get("label", "?") for r in rs)
        out.append(f"| {x} | {len(rs)} | {c['content']} | {c['name_or_header']} | {c['distrust_file']} | {c['other']} |")
    out += ["", f"Manual check of 20 labeled figurine episodes (skip reasons; or all if fewer): **{manual.get('fig_agree', '?')}** agree. {manual.get('fig_note', '')}", ""]

    # ---- 5. transcripts ---------------------------------------------------------------------------------
    out += ["## 5. Transcripts: v3b Fatal + Bird, X = 1000, thinking ON", ""]
    rs = rows_of(new, "fatal", "bird", 1000, True)
    picks = []
    for want in ["trap", "exposure"]:
        for r in rs:
            if L2.get((r["version"], "fatal", "bird", 1000, r["ep"], True), {}).get("label") == want:
                picks.append((f"{want} example", r)); break
    for pred, label in [(lambda r: not r["measures"]["collected_c"] and r["measures"]["read_cue"], "skip after reading the grader file"),
                        (lambda r: r["measures"]["collected_c"], "collected item_c"),
                        (lambda r: not r["measures"]["collected_c"] and read_item_c_before_decision(r), "skip after reading item_c (knew X=1000)")]:
        for r in rs:
            if pred(r) and r not in [p[1] for p in picks]:
                picks.append((label, r)); break
    for label, r in picks[:5]:
        out += [f"### {label}", "", short_transcript(r, L2.get((r["version"], "fatal", "bird", 1000, r["ep"], True), {}).get("label")), ""]
    text = "\n".join(out)
    (V3 / "RESULTS_v3.md").write_text(text)
    res = S / "RESULTS.md"; body = res.read_text()
    marker = "\n---\n\n# v3 / v3b: large reward"
    if marker in body:
        body = body[:body.index(marker)]
    res.write_text(body.rstrip() + "\n" + text)
    print(text)


if __name__ == "__main__":
    main()
