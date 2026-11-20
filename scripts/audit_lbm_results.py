"""Independent recomputation of the headline numbers in the LBM law-study READMEs.

Reads only the committed per-seed JSONs under docs/results/ and prints the figures the
READMEs quote, so each can be compared by eye or by `--check`. It does not import the
study code or the authors' analysis scripts.

Usage: python scripts/audit_lbm_results.py [--root docs/results]
"""

from __future__ import annotations

import argparse
import itertools
import json
from functools import partial
from pathlib import Path

import numpy as np
from scipy import stats

LIVE = {
    "A": "seeds0-14_A_legacy.json",
    "B": "seeds0-14_B_live_nocoupling.json",
    "C": "seeds0-14_C_mono_0.5.json",
    "D": "seeds0-14_D_dipole_0.5.json",
    "E": "seeds0-14_E_mono_2.0.json",
    "F": "seeds0-14_F_dipole_2.0.json",
}
PROFILE = {
    "uniform": "seeds0-14_uniform.json",
    "lbm-vectors": "seeds0-14_lbm.json",
    "lbm-geometry": "seeds0-14_lbm_geometry.json",
    "lbm-channel": "seeds0-14_lbm_channel_geometry.json",
    "lbm-geometry-floor": "seeds0-14_lbm_geometry_flow_floor.json",
}
SWEEP_DRAGS = {
    ("monopole", 0.75): "seeds0-14_monopole_0.75.json",
    ("monopole", 1.0): "seeds0-14_monopole_1.0.json",
    ("monopole", 1.25): "seeds0-14_monopole_1.25.json",
    ("monopole", 1.5): "seeds0-14_monopole_1.5.json",
    ("dipole", 1.0): "seeds0-14_dipole_1.0.json",
    ("dipole", 1.5): "seeds0-14_dipole_1.5.json",
}
MARGIN, REV = 0.2, 0.2


def rows(path: Path) -> list[dict]:
    r = sorted(json.loads(path.read_text()), key=lambda x: x["seed"])
    assert [x["seed"] for x in r] == list(range(15)), path
    return r


def col(r, key):
    return np.array([x[key] for x in r], float)


def passed(r):
    return np.array([x["min_delta"] >= MARGIN and x["min_reversed"] >= REV for x in r])


def summarize(name, r):
    p = passed(r)
    flagged = np.array([x["passed"] for x in r])
    fails = [
        (x["seed"], round(x["min_delta"], 3), round(x["min_reversed"], 3))
        for x, ok in zip(r, p, strict=True)
        if not ok
    ]
    note = "" if (p == flagged).all() else "  !! recomputed gate differs from stored `passed`"
    print(
        f"  {name:22s} {p.sum():2d}/15  median min_delta {np.median(col(r, 'min_delta')):.3f}"
        f"  fails {fails}{note}"
    )


def _mean_of(rows_, key):
    return float(np.mean([x[key] for x in rows_]))


def wilcoxon(a, b):
    d = a - b
    res = stats.wilcoxon(d)
    return d.mean(), int((d < 0).sum()), int((d > 0).sum()), res.pvalue


def fisher(p1, p2):
    return stats.fisher_exact([[p1.sum(), 15 - p1.sum()], [p2.sum(), 15 - p2.sum()]])[1]


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, run)
    return adj


def profile(root: Path):
    d = root / "embodied_law_lbm_profile"
    R = {k: rows(d / v) for k, v in PROFILE.items()}
    print("== profile: pass / median / failing seeds (min_delta, min_reversed)")
    for k, r in R.items():
        summarize(k, r)
    print("== profile: Fisher exact on pass counts")
    for a, b in itertools.combinations(R, 2):
        print(f"  {a} vs {b}: p = {fisher(passed(R[a]), passed(R[b])):.3f}")
    first5 = {
        k: [x for x in r if x["seed"] < 5] for k, r in R.items() if k in ("uniform", "lbm-vectors")
    }
    print("== profile: first five seeds (min_delta / min_reversed)")
    for u, v in zip(first5["uniform"], first5["lbm-vectors"], strict=True):
        print(
            f"  seed {u['seed']}: uniform {u['min_delta']:.3f}/{u['min_reversed']:.3f}"
            f"  lbm {v['min_delta']:.3f}/{v['min_reversed']:.3f}"
        )
    print("== profile: margins quoted in the text")
    g = {x["seed"]: x for x in R["lbm-geometry"]}
    for s in (1, 7, 9, 13):
        print(
            f"  geometry seed {s}: margin shortfall {MARGIN - g[s]['min_delta']:+.3f},"
            f" reversed shortfall {REV - g[s]['min_reversed']:+.3f}"
        )
    v4 = {x["seed"]: x for x in R["lbm-vectors"]}[4]
    print(f"  vectors seed 4 margin shortfall {MARGIN - v4['min_delta']:+.3f}")
    print("== profile: flow-floor failing seeds in full")
    for x in R["lbm-geometry-floor"]:
        if not (x["min_delta"] >= MARGIN and x["min_reversed"] >= REV):
            print(f"  seed {x['seed']}: {x['min_delta']:.3f} / {x['min_reversed']:.3f}")
    names = ("lbm-vectors", "lbm-geometry", "lbm-channel", "lbm-geometry-floor")
    res = [wilcoxon(col(R[n], "min_delta"), col(R["uniform"], "min_delta")) for n in names]
    for n, (m, w, b, p), h in zip(names, res, holm(np.array([r[3] for r in res])), strict=True):
        print(
            f"  paired Wilcoxon min_delta {n} vs uniform: mean {m:+.3f} worse/better {w}/{b}"
            f" p={p:.4f} Holm(4)={h:.4f}"
        )
    ref = root / "embodied_law_flow_head" / "summary.json"
    if ref.exists():
        old = {r["seed"]: r for r in json.loads(ref.read_text())["results"]}
        diff = max(
            abs(old[s]["min_delta"] - x["min_delta"]) for s, x in enumerate(R["uniform"][:5])
        )
        print(f"  uniform seeds 0-4 vs embodied_law_flow_head: max |min_delta diff| {diff:.1e}")
    return R


def live(root: Path):
    d = root / "embodied_law_lbm_live"
    R = {k: rows(d / v) for k, v in LIVE.items()}
    print("== live: pass / median / failing seeds")
    for k, r in R.items():
        summarize(k, r)
    print(
        "== live: A vs B max |min_delta| difference:",
        float(np.abs(col(R["A"], "min_delta") - col(R["B"], "min_delta")).max()),
    )
    print("== live: Fisher exact, all pairs")
    for a, b in itertools.combinations(R, 2):
        print(f"  {a} vs {b}: p = {fisher(passed(R[a]), passed(R[b])):.3f}")
    print("== live: paired Wilcoxon on min_delta against A")
    ps = {}
    for k in "CDEF":
        m, w, b, p = wilcoxon(col(R[k], "min_delta"), col(R["A"], "min_delta"))
        ps[k] = p
        print(f"  {k}: mean {m:+.3f} worse/better {w}/{b} p={p:.4f}")
    print(f"  E raw p {ps['E']:.5f}; Bonferroni x4 = {min(1, 4 * ps['E']):.4f}")
    print("== live: monopole vs dipole at matched drag")
    for mono, dip in (("E", "F"), ("C", "D")):
        m, w, b, p = wilcoxon(col(R[dip], "min_delta"), col(R[mono], "min_delta"))
        print(f"  {dip}-{mono}: mean {m:+.3f} p={p:.4f}")
    print("== live: min_reversed change against A")
    for k in "BCDEF":
        m, w, b, p = wilcoxon(col(R[k], "min_reversed"), col(R["A"], "min_reversed"))
        print(f"  {k}: mean {m:+.3f} p={p:.3f}")
    return R


def sweep(root: Path, L):
    d = root / "embodied_law_lbm_live" / "drag_sweep"
    R = {
        ("monopole", 0.0): L["B"],
        ("monopole", 0.5): L["C"],
        ("monopole", 2.0): L["E"],
        ("dipole", 0.0): L["B"],
        ("dipole", 0.5): L["D"],
        ("dipole", 2.0): L["F"],
    }
    for k, v in SWEEP_DRAGS.items():
        R[k] = rows(d / v)
    print("== sweep: per condition (change vs B)")
    mono = sorted(x for x in R if x[0] == "monopole")
    base = col(R[("monopole", 0.0)], "min_delta")
    raw = []
    for k in sorted(R, key=lambda t: (t[0], t[1])):
        if k[1] == 0.0:
            continue
        r = R[k]
        m, w, b, p = wilcoxon(col(r, "min_delta"), base)
        if k[0] == "monopole":
            raw.append(p)
        summarize(f"{k[0]} {k[1]}", r)
        print(f"{'':25s}mean change {m:+.3f} worse/better {w}/{b} p={p:.4f}")
    hp = holm(np.array(raw))
    print(
        "  Holm over monopole drags:",
        dict(zip([k[1] for k in mono if k[1] > 0], hp.round(3), strict=True)),
    )
    drags = np.array([k[1] for k in mono])
    Y = np.stack([col(R[k], "min_delta") for k in mono], axis=1)
    slopes = np.array([np.polyfit(drags, y, 1)[0] for y in Y])
    print(
        f"== sweep: per-seed slope mean {slopes.mean():+.4f}, negative on {(slopes < 0).sum()}/15,"
        f" one-sided Wilcoxon p = {stats.wilcoxon(slopes, alternative='less').pvalue:.4f}"
    )
    rho, p = stats.spearmanr(np.repeat(drags[None, :], 15, 0).ravel(), Y.ravel())
    print(f"  pooled Spearman rho {rho:+.3f} p={p:.4f}")
    mean_change = (Y - Y[:, :1]).mean(0)
    for label, sel in (
        ("seven drags incl. 0", slice(0, None)),
        ("six drags excl. 0", slice(1, None)),
    ):
        x, y = drags[sel], mean_change[sel]
        lin = np.polyfit(x, y, 1)
        lres = float(((np.polyval(lin, x) - y) ** 2).sum())
        best = None
        for t in np.linspace(0.25, 1.75, 61):
            A = np.stack([np.maximum(x - t, 0), np.ones(len(x))], 1)
            sse = float(((A @ np.linalg.lstsq(A, y, rcond=None)[0] - y) ** 2).sum())
            best = (sse, t) if best is None or sse < best[0] else best
        print(
            f"  {label}: linear SSE {lres:.5f}, best hinge SSE {best[0]:.5f} at onset {best[1]:.2f}"
        )
    rng = np.random.default_rng(0)
    print("  bootstrap 95% CI of paired mean change (seed resample, rng seed 0):")
    for j, k in enumerate(mono):
        if k[1] == 0:
            continue
        diffs = Y[:, j] - Y[:, 0]
        boot = [diffs[rng.integers(0, 15, 15)].mean() for _ in range(5000)]
        print(
            f"    drag {k[1]}: [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]"
        )
    print("== sweep: dipole - monopole at matched drag")
    raws = []
    out = []
    for dr in (0.5, 1.0, 1.5, 2.0):
        m, w, b, p = wilcoxon(
            col(R[("dipole", dr)], "min_delta"), col(R[("monopole", dr)], "min_delta")
        )
        raws.append(p)
        out.append((dr, m, p))
    for (dr, m, p), hpv in zip(out, holm(np.array(raws)), strict=True):
        print(f"  drag {dr}: mean {m:+.3f} p={p:.4f} Holm(4) {hpv:.4f}")
    for k in sorted(R, key=lambda t: (t[0], t[1])):
        if k[0] == "dipole" and k[1] > 0:
            m, w, b, p = wilcoxon(col(R[k], "min_delta"), base)
            print(f"  dipole {k[1]} vs B: mean {m:+.3f} p={p:.3f}")
    rev = [col(R[k], "min_reversed").mean() for k in R if k[1] > 0]
    print(
        f"  min_reversed means across coupled conditions: {min(rev):.2f}..{max(rev):.2f},"
        f" B {col(R[('monopole', 0.0)], 'min_reversed').mean():.2f}"
    )


def estimate(root: Path):
    d = json.loads(
        (root / "embodied_law_lbm_live" / "estimate_error" / "per_seed.json").read_text()
    )
    S = {k: sorted(v, key=lambda x: x["seed"]) for k, v in d["same_fluid"].items()}
    print(
        "== estimate error: means over seeds (angle_mean, frac_over_45, law_angle_mean,"
        " magnitude_dev, median min_delta)"
    )
    for k, r in S.items():
        a = partial(_mean_of, r)
        print(
            f"  {k}: {a('angle_mean'):.2f}  {a('frac_over_45'):.3f}  {a('law_angle_mean'):.2f}"
            f"  {a('magnitude_dev_mean'):.3f}  {np.median([x['min_delta'] for x in r]):.3f}"
        )
    B = np.array([x["angle_mean"] for x in S["B"]])
    for k in "CDEF":
        v = np.array([x["angle_mean"] for x in S[k]])
        print(
            f"  {k} vs B angle: higher on {(v > B).sum()}/15, p={stats.wilcoxon(v - B).pvalue:.1e}"
        )
    ang = np.concatenate([[x["angle_mean"] for x in S[k]] for k in "ABCDEF"])
    md = np.concatenate([[x["min_delta"] for x in S[k]] for k in "ABCDEF"])
    cond = np.repeat(np.arange(6), 15)
    print(
        "  pooled Spearman (n=90): "
        f"{stats.spearmanr(ang, md)[0]:+.2f} p={stats.spearmanr(ang, md)[1]:.2f}"
    )

    def within(mask):
        a, m, c = ang[mask], md[mask], cond[mask]
        ar = a - np.array([a[c == u].mean() for u in c])
        mr = m - np.array([m[c == u].mean() for u in c])
        return stats.spearmanr(ar, mr)

    r1 = within(np.ones(90, bool))
    r2 = within(cond >= 2)
    print(
        f"  within-condition pooled: all {r1[0]:+.2f} p={r1[1]:.3f};"
        f" coupled {r2[0]:+.2f} p={r2[1]:.3f}"
    )
    for i, k in enumerate("ABCDEF"):
        if k in "CDEF":
            sl = slice(15 * i, 15 * (i + 1))
            rr = stats.spearmanr(ang[sl], md[sl])
            print(f"    {k}: rho {rr[0]:+.2f} p={rr[1]:.3f}")
    print("== estimate error: mediation point estimates (B pooled with E or F, OLS)")
    for med in ("angle_mean", "law_angle_mean", "magnitude_dev_mean"):
        for k in "EF":
            y = np.array([x["min_delta"] for x in S["B"] + S[k]])
            mv = np.array([x[med] for x in S["B"] + S[k]])
            t = np.r_[np.zeros(15), np.ones(15)]
            tot = np.linalg.lstsq(np.c_[np.ones(30), t], y, rcond=None)[0][1]
            direct = np.linalg.lstsq(np.c_[np.ones(30), t, mv], y, rcond=None)[0][1]
            print(
                f"  {med:20s} {k}: total {tot:+.3f} direct {direct:+.3f}"
                f" indirect {tot - direct:+.3f}"
            )
    print("== estimate error: cross-evaluation (median min_delta, mean angle, vs B->B)")
    base = np.array([x["min_delta"] for x in S["B"]])
    for name, r in {**{f"{k}->{k}": S[k] for k in "EF"}, **d["cross_eval"]}.items():
        r = sorted(r, key=lambda x: x["seed"])
        v = np.array([x["min_delta"] for x in r])
        print(
            f"  {name}: median {np.median(v):.3f} angle {np.mean([x['angle_mean'] for x in r]):.2f}"
            f"  diff {np.mean(v - base):+.3f} p={stats.wilcoxon(v - base).pvalue:.3f}"
            f" worse {(v < base).sum()}/15"
        )
    return S


def mechanism(root: Path):
    d = root / "embodied_law_lbm_live" / "mechanism"
    seeds = {}
    for f in sorted((d / "eval_per_seed").glob("seed*.json")):
        j = json.loads(f.read_text())
        seeds[j["seed"]] = j["rows"]
    assert sorted(seeds) == list(range(15))
    keys = ("official", "law", "noself", "selfonly", "noself_sum")

    def align(fluid, genomes, key):
        return np.array(
            [
                np.nanmean(
                    [r[key] for r in seeds[s] if r["fluid"] == fluid and r["genomes"] == genomes]
                )
                for s in range(15)
            ]
        )

    def min_delta(fluid, genomes, key):
        out = []
        for s in range(15):
            ds = []
            for dr in sorted({r["direction"] for r in seeds[s]}):
                g = [
                    r[key]
                    for r in seeds[s]
                    if r["fluid"] == fluid and r["genomes"] == genomes and r["direction"] == dr
                ]
                i = [
                    r[key]
                    for r in seeds[s]
                    if r["fluid"] == fluid and r["genomes"] == "init" and r["direction"] == dr
                ]
                ds.append(np.mean(g) - np.mean(i))
            out.append(min(ds))
        return np.array(out)

    print("== mechanism: mean alignment")
    for fluid in "BEF":
        for g in ("B", "init"):
            print(
                f"  {fluid} {g:5s}" + "".join(f" {k}={align(fluid, g, k).mean():.3f}" for k in keys)
            )
    print("== mechanism: min_delta change in E (and F) against B")
    rng = np.random.default_rng(0)
    fam = {}
    for fluid in "EF":
        for g in ("B", "E"):
            for key in keys:
                a, b = min_delta(fluid, g, key), min_delta("B", g, key)
                dlt = a - b
                p = stats.wilcoxon(dlt).pvalue if np.any(dlt != 0) else 1.0
                boot = [dlt[rng.integers(0, 15, 15)].mean() for _ in range(3000)]
                tag = ""
                if fluid == "E" and key in ("official", "law", "noself", "selfonly"):
                    fam[(g, key)] = p
                print(
                    f"  {g}-trained in {fluid}, {key:10s}: median {np.median(a):.3f}"
                    f" change {dlt.mean():+.3f}"
                    f" [{np.percentile(boot, 2.5):+.3f},{np.percentile(boot, 97.5):+.3f}] p={p:.4f}"
                    f" worse/better {(dlt < 0).sum()}/{(dlt > 0).sum()}{tag}"
                )
    ks = list(fam)
    hp = holm(np.array([fam[k] for k in ks]))
    print(
        "  Holm over the 8 fluid-E tests:",
        {f"{g}/{k}": round(float(h), 4) for (g, k), h in zip(ks, hp, strict=True)},
    )
    for g in ("B", "E"):
        tot = min_delta("E", g, "official") - min_delta("B", g, "official")
        no = min_delta("E", g, "noself") - min_delta("B", g, "noself")
        boot = []
        for _ in range(3000):
            idx = rng.integers(0, 15, 15)
            t = tot[idx].mean()
            boot.append(1 - no[idx].mean() / t if abs(t) > 1e-9 else np.nan)
        print(
            f"  share removed {g}-trained: {1 - no.mean() / tot.mean():.2f}"
            f" [{np.nanpercentile(boot, 2.5):.2f}, {np.nanpercentile(boot, 97.5):.2f}]"
        )
    print("== mechanism: cos(thrust, own wake), fluid E")
    for g in ("B", "init"):
        v = [
            r["cos_thrust_own_wake"]
            for s in range(15)
            for r in seeds[s]
            if r["fluid"] == "E" and r["genomes"] == g
        ]
        print(f"  {g}: {np.nanmean(v):.2f}")
    print("== mechanism: wake statistics by fluid / genomes")
    wk = (
        "cos_thrust_own_wake",
        "own_wake_mean",
        "others_wake_mean",
        "total_perturbation_mean",
        "superposition_residual_mean",
        "superposition_residual_max",
    )
    for fluid in "EF":
        for g in ("B", "init"):
            v = {
                k: np.nanmean(
                    [
                        r[k]
                        for s in range(15)
                        for r in seeds[s]
                        if r["fluid"] == fluid and r["genomes"] == g
                    ]
                )
                for k in wk
            }
            print(f"  {fluid} {g:5s}" + "".join(f" {k}={x:.3f}" for k, x in v.items()))
    rep = {}
    for f in sorted((d / "replay_per_seed").glob("seed*.json")):
        j = json.loads(f.read_text())
        rep[j["seed"]] = j["rows"]
    print("== mechanism: cross-replay (official alignment, own vs foreign wake)")
    for g in ("B", "init"):
        own, frn = [], []
        for s in range(15):
            own.append(
                np.mean(
                    [r["official"] for r in rep[s] if r["genomes"] == g and r["series_from"] == g]
                )
            )
            frn.append(
                np.mean(
                    [r["official"] for r in rep[s] if r["genomes"] == g and r["series_from"] != g]
                )
            )
        own, frn = np.array(own), np.array(frn)
        print(
            f"  {g}: own {own.mean():.3f} foreign {frn.mean():.3f} diff {(own - frn).mean():+.3f}"
            f" p={stats.wilcoxon(own - frn).pvalue:.4f}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="docs/results")
    root = Path(ap.parse_args().root)
    profile(root)
    L = live(root)
    sweep(root, L)
    estimate(root)
    mechanism(root)


if __name__ == "__main__":
    main()
