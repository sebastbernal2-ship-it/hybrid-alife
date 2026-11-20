"""Drag-sweep analysis (needs numpy, scipy). Run: python analysis.py.

Prints every statistic quoted in README.md.
"""

import json
import pathlib

import numpy as np
from scipy.stats import spearmanr, wilcoxon

HERE = pathlib.Path(__file__).resolve().parent
R = str(HERE.parent) + "/"
OUT = str(HERE) + "/"


def load(path):
    d = json.load(open(path))
    if isinstance(d, dict):
        d = d["results"]
    return {r["seed"]: r for r in d}


base = load(R + "seeds0-14_B_live_nocoupling.json")
conds = {
    ("monopole", 0.0): base,
    ("monopole", 0.5): load(R + "seeds0-14_C_mono_0.5.json"),
    ("monopole", 2.0): load(R + "seeds0-14_E_mono_2.0.json"),
    ("dipole", 0.5): load(R + "seeds0-14_D_dipole_0.5.json"),
    ("dipole", 2.0): load(R + "seeds0-14_F_dipole_2.0.json"),
}
for mode, drags in (("monopole", (0.75, 1.0, 1.25, 1.5)), ("dipole", (1.0, 1.5))):
    for d in drags:
        rows = {}
        conds[(mode, d)] = load(f"{OUT}seeds0-14_{mode}_{d}.json")
# reproducibility
rep = load(f"{OUT}reproducibility_monopole_2.0_seeds0-2.json")
E = conds[("monopole", 2.0)]
print(
    "REPRO max |min_delta diff| seeds0-2:",
    max(abs(rep[s]["min_delta"] - E[s]["min_delta"]) for s in range(3)),
    "max |min_reversed diff|:",
    max(abs(rep[s]["min_reversed"] - E[s]["min_reversed"]) for s in range(3)),
    "passed same:",
    all(rep[s]["passed"] == E[s]["passed"] for s in range(3)),
)
seeds = range(15)


def md(c):
    return np.array([conds[c][s]["min_delta"] for s in seeds])


b = md(("monopole", 0.0))
out = {}
print("cond pass median fails wilcoxon_p meanchange worse/better")
for c in sorted(conds, key=lambda k: (k[0], k[1])):
    x = md(c)
    fails = [(s, round(conds[c][s]["min_delta"], 3)) for s in seeds if not conds[c][s]["passed"]]
    p = wilcoxon(x, b).pvalue if c != ("monopole", 0.0) else float("nan")
    ch = (x - b).mean()
    w = int((x < b).sum())
    bt = int((x > b).sum())
    print(
        c,
        f"{sum(conds[c][s]['passed'] for s in seeds)}/15",
        round(float(np.median(x)), 3),
        fails,
        round(p, 4),
        round(ch, 3),
        f"{w}/{bt}",
    )
    out[f"{c[0]}_{c[1]}"] = {
        "pass": sum(conds[c][s]["passed"] for s in seeds),
        "median": float(np.median(x)),
        "fails": fails,
        "p_vs_B": None if np.isnan(p) else float(p),
        "mean_change": float(ch),
        "worse": w,
        "better": bt,
    }
# trend
mono = [0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
X = np.concatenate([[d] * 15 for d in mono])
Y = np.concatenate([md(("monopole", d)) for d in mono])
rho, p = spearmanr(X, Y)
print("monopole Spearman pooled rho,p", rho, p)
# paired trend: per-seed Spearman of change vs drag, then test mean slope
slopes = [
    np.polyfit(mono, [conds[("monopole", d)][s]["min_delta"] for d in mono], 1)[0] for s in seeds
]
print(
    "per-seed slope mean",
    np.mean(slopes),
    "wilcoxon p (slopes<0)",
    wilcoxon(slopes, alternative="less").pvalue,
    "n neg",
    sum(1 for s in slopes if s < 0),
)
# Jonckheere-style: sum of Kendall tau between drag and min_delta within seed
# onset: paired effect vs B per drag, Holm over 6 coupled monopole drags
ps = {d: wilcoxon(md(("monopole", d)), b).pvalue for d in mono[1:]}
order = sorted(ps, key=ps.get)
m = len(order)
holm = {}
run = 0
for i, d in enumerate(order):
    run = max(run, min(1, (m - i) * ps[d]))
    holm[d] = run
print("monopole vs B p raw / Holm(6):", {d: (round(ps[d], 4), round(holm[d], 4)) for d in mono[1:]})
# dipole
dip = [0.5, 1.0, 1.5, 2.0]
psd = {d: wilcoxon(md(("dipole", d)), b).pvalue for d in dip}
print("dipole vs B p:", {d: round(v, 4) for d, v in psd.items()})
for d in (0.5, 1.0, 1.5, 2.0):
    print(
        "mono vs dipole at",
        d,
        "paired diff mean(dip-mono)",
        round(float((md(("dipole", d)) - md(("monopole", d))).mean()), 3),
        "p",
        round(wilcoxon(md(("dipole", d)), md(("monopole", d))).pvalue, 4),
    )
# bootstrap CI of paired mean change per monopole drag + isotonic-ish onset
rng = np.random.default_rng(0)
print("paired mean change (95% bootstrap CI) monopole:")
for d in mono[1:]:
    ch = md(("monopole", d)) - b
    bs = [rng.choice(ch, 15).mean() for _ in range(5000)]
    print(d, round(ch.mean(), 3), np.round(np.percentile(bs, [2.5, 97.5]), 3))
print(
    "min_reversed means:",
    {
        f"{c[0]}{c[1]}": round(float(np.mean([conds[c][s]["min_reversed"] for s in seeds])), 3)
        for c in conds
    },
)
# linear vs threshold fit on mean change
means = np.array([(md(("monopole", d)) - b).mean() for d in mono])
lin = np.polyfit(mono, means, 1)
lres = ((np.polyval(lin, mono) - means) ** 2).sum()
best = None
for t in np.linspace(0.25, 1.75, 61):
    A = np.stack([np.maximum(np.array(mono) - t, 0), np.ones(len(mono))], 1)
    co, res, _, _ = np.linalg.lstsq(A, means, rcond=None)
    r = ((A @ co - means) ** 2).sum()
    if best is None or r < best[0]:
        best = (r, t, co)
print("linear SSE", lres, "hinge SSE", best[0], "hinge onset", best[1], "slope", best[2][0])
json.dump(out, open(OUT + "summary.json", "w"), indent=1)
