"""Summarize Test 1: per-format effect of the warning (B - A), French history vs controls,
with item-level bootstrap 95% CIs and a difference-in-differences (FH effect - control effect)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rng = np.random.default_rng(0)


def boot(x, n=5000):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    if len(x) == 0:
        return (np.nan, np.nan, np.nan)
    m = rng.choice(x, (n, len(x))).mean(1)
    return (x.mean(), *np.percentile(m, [2.5, 97.5]))


def did(fh, ct, n=5000):
    fh, ct = np.asarray(fh, float), np.asarray(ct, float)
    m = rng.choice(fh, (n, len(fh))).mean(1) - rng.choice(ct, (n, len(ct))).mean(1)
    return (fh.mean() - ct.mean(), *np.percentile(m, [2.5, 97.5]))


def fmt(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def summarize(tag):
    d = pd.read_json(ROOT / "results/test1" / f"{tag}.jsonl", lines=True)
    w = d.pivot_table(index=["id", "subdomain", "fmt"], columns="cond", values="reliance_off").reset_index()
    w["delta"] = w["B"] - w["A"]
    lines = [f"### {tag}", "",
             "| format | subdomain | A | B | B−A [95% CI] |", "|---|---|---|---|---|"]
    for f in ["abstain", "lookup", "conflict", "conflict_ind"]:
        for s in ["french_history", "other_history", "french_nonhistory", "general"]:
            x = w[(w.fmt == f) & (w.subdomain == s)]
            lines.append(f"| {f} | {s} | {x.A.mean():.3f} | {x.B.mean():.3f} | {fmt(boot(x.delta))} |")
    lines += ["", "| format | DiD: FH − all controls [95% CI] | DiD: FH − other_history |", "|---|---|---|"]
    for f in ["abstain", "lookup", "conflict", "conflict_ind"]:
        fh = w[(w.fmt == f) & (w.subdomain == "french_history")].delta
        ct = w[(w.fmt == f) & (w.subdomain != "french_history")].delta
        oh = w[(w.fmt == f) & (w.subdomain == "other_history")].delta
        lines.append(f"| {f} | {fmt(did(fh, ct))} | {fmt(did(fh, oh))} |")
    acc = d[d.fmt == "abstain"].groupby(["domain", "cond"]).acc_given_answer.mean().unstack()
    lines += ["", "Accuracy among non-abstain mass (abstain format): " +
              "; ".join(f"{dom}: A={r.A:.3f} B={r.B:.3f}" for dom, r in acc.iterrows()), ""]
    return "\n".join(lines)




def compare(base_tag, ft_tag, cond="A"):
    """Training effect: fine-tuned minus base, same condition, paired by item."""
    def load(t):
        d = pd.read_json(ROOT / "results/test1" / f"{t}.jsonl", lines=True)
        return d[d.cond == cond].set_index(["id", "fmt"])[["subdomain", "reliance_off"]]
    b, f = load(base_tag), load(ft_tag)
    w = b.join(f[["reliance_off"]], rsuffix="_ft").reset_index()
    w["delta"] = w["reliance_off_ft"] - w["reliance_off"]
    lines = [f"### training effect: {ft_tag} − {base_tag} (condition {cond})", "",
             "| format | French history | other history | French non-history | general | DiD FH − all controls [95% CI] |",
             "|---|---|---|---|---|---|"]
    for fm in ["abstain", "lookup", "conflict", "conflict_ind"]:
        x = w[w.fmt == fm]
        cells = [f"{x[x.subdomain == s].reliance_off.mean():.2f}→{x[x.subdomain == s].reliance_off_ft.mean():.2f}"
                 for s in ["french_history", "other_history", "french_nonhistory", "general"]]
        fh, ct = x[x.subdomain == "french_history"].delta, x[x.subdomain != "french_history"].delta
        lines.append(f"| {fm} | " + " | ".join(cells) + f" | {fmt(did(fh, ct))} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    if sys.argv[1] == "--compare":
        print(compare(sys.argv[2], sys.argv[3]))
    else:
        print("\n".join(summarize(t) for t in sys.argv[1:]))
