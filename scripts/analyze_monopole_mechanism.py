"""Statistics for the monopole mechanism study (reads the per-seed JSONs from the measure script).

Usage: python scripts/analyze_monopole_mechanism.py --eval DIR [--replay DIR] [--json OUT]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

VARIANTS = ("official", "law", "noself", "selfonly", "noself_sum")
SEEDS = range(15)


def load(directory: Path) -> list[dict]:
    return [json.loads((directory / f"seed{s}.json").read_text()) for s in SEEDS]


def table(eval_rows: list[dict]) -> dict:
    """alignment[fluid][genomes][variant] -> [seed] mean over directions; min_delta likewise."""
    out: dict = {}
    for fluid in ("B", "E", "F"):
        out[fluid] = {}
        for g in ("B", "E", "init"):
            out[fluid][g] = {}
            for v in VARIANTS:
                per_seed = []
                for result in eval_rows:
                    rows = [r for r in result["rows"] if r["fluid"] == fluid and r["genomes"] == g]
                    per_seed.append([r[v] for r in rows])  # 4 directions
                out[fluid][g][v] = np.array(per_seed)  # [seed, direction]
    return out


def min_delta(tab, fluid, g, v):
    """Per-seed min over directions of (evolved - initial), both in the same fluid and variant."""
    return (tab[fluid][g][v] - tab[fluid]["init"][v]).min(axis=1)


def bootstrap_ci(x, n=5000, seed=0):
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(x, len(x)).mean() for _ in range(n)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a, b):
    d = a - b
    try:
        p = stats.wilcoxon(a, b).pvalue
    except ValueError:
        p = float("nan")
    lo, hi = bootstrap_ci(d)
    return {"mean_diff": float(d.mean()), "ci": [lo, hi], "wilcoxon_p": float(p),
            "worse": int((d < 0).sum()), "better": int((d > 0).sum())}


def holm(ps):
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", type=Path, required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    tab = table(load(args.eval))
    report: dict = {}

    print("== reproduction: median min_delta (official score)")
    for fluid, g in (("B", "B"), ("E", "B"), ("E", "E"), ("F", "B"), ("F", "E")):
        median = np.median(min_delta(tab, fluid, g, "official"))
        print(f"  genomes {g} in fluid {fluid}: {median:.3f}")

    print("\n== mean alignment (over directions and seeds)")
    print("fluid genomes " + " ".join(f"{v:>10}" for v in VARIANTS))
    report["alignment"] = {}
    for fluid in ("B", "E", "F"):
        for g in ("B", "init"):
            row = [float(tab[fluid][g][v].mean()) for v in VARIANTS]
            report["alignment"][f"{fluid}/{g}"] = dict(zip(VARIANTS, row, strict=True))
            print(f"{fluid:>5} {g:>7} " + " ".join(f"{x:10.3f}" for x in row))

    print("\n== min_delta by scoring variant, B-trained genomes (median; mean change vs fluid B,"
          " [95% CI], Wilcoxon p, seeds worse/better)")
    report["min_delta"] = {}
    for g in ("B", "E"):
        for v in VARIANTS:
            ref = min_delta(tab, "B", g, "official")
            for fluid in ("E", "F"):
                d = min_delta(tab, fluid, g, v)
                res = paired(d, ref)
                report["min_delta"][f"{g}-trained/{fluid}/{v}"] = {
                    "median": float(np.median(d)), **res}
                print(f"  {g}-trained in {fluid}, scored {v:>10}: median {np.median(d):6.3f} "
                      f"change {res['mean_diff']:+.3f} [{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] "
                      f"p={res['wilcoxon_p']:.4f} worse/better {res['worse']}/{res['better']}")

    # ---- Holm over the primary family: fluid E, both genome sets, four scoring variants -----
    family = [(g, v) for g in ("B", "E") for v in ("official", "law", "noself", "selfonly")]
    ps = [report["min_delta"][f"{g}-trained/E/{v}"]["wilcoxon_p"] for g, v in family]
    for (g, v), adj in zip(family, holm(ps), strict=True):
        report["min_delta"][f"{g}-trained/E/{v}"]["holm_p_8"] = adj
    print("\n== Holm-adjusted p over the 8 fluid-E tests (2 genome sets x 4 scorings)")
    for (g, v), adj in zip(family, holm(ps), strict=True):
        raw = report["min_delta"][f"{g}-trained/E/{v}"]["wilcoxon_p"]
        print(f"  {g}-trained, {v:>8}: raw {raw:.4f} Holm {adj:.4f}")

    # ---- Which side of min_delta moves: evolved or initial alignment ---------------------
    print("\n== alignment change in fluid E vs fluid B (B-trained evolved vs initial), per scoring")
    report["side"] = {}
    for v in ("official", "law", "noself"):
        ev = (tab["E"]["B"][v] - tab["B"]["B"]["official"]).mean(axis=1)
        ini = (tab["E"]["init"][v] - tab["B"]["init"]["official"]).mean(axis=1)
        pe, pi = paired(ev, np.zeros_like(ev)), paired(ini, np.zeros_like(ini))
        report["side"][v] = {"evolved": pe, "initial": pi}
        print(f"  {v:>8}: evolved {pe['mean_diff']:+.3f} [{pe['ci'][0]:+.3f},{pe['ci'][1]:+.3f}] "
              f"p={pe['wilcoxon_p']:.4f} | initial {pi['mean_diff']:+.3f} "
              f"[{pi['ci'][0]:+.3f},{pi['ci'][1]:+.3f}] p={pi['wilcoxon_p']:.4f}")

    # ---- Share of the min_delta loss removed by taking the own wake out of the target ----
    print("\n== share of the fluid-E loss removed by scoring without the own wake (seed bootstrap)")
    report["explained"] = {}
    rng = np.random.default_rng(1)
    for g in ("B", "E"):
        ref = min_delta(tab, "B", g, "official")
        loss_official = ref - min_delta(tab, "E", g, "official")
        loss_noself = ref - min_delta(tab, "E", g, "noself")
        shares = []
        for _ in range(5000):
            idx = rng.integers(0, len(ref), len(ref))
            lo = loss_official[idx].mean()
            shares.append(1.0 - loss_noself[idx].mean() / lo if abs(lo) > 1e-9 else np.nan)
        point = 1.0 - loss_noself.mean() / loss_official.mean()
        ci = np.nanpercentile(shares, [2.5, 97.5])
        report["explained"][g] = {"share": float(point), "ci": [float(ci[0]), float(ci[1])]}
        print(f"  {g}-trained: {point:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]")

    print("\n== wake statistics (fluid E, B-trained evolved / initial)")
    report["wake"] = {}
    for g in ("B", "init"):
        rows = [r for res in load(args.eval) for r in res["rows"]
                if r["fluid"] == "E" and r["genomes"] == g]
        keys = ("own_wake_mean", "others_wake_mean", "total_perturbation_mean",
                "cos_thrust_own_wake", "superposition_residual_mean", "superposition_residual_max")
        report["wake"][g] = {k: float(np.nanmean([r[k] for r in rows])) for k in keys}
        print(f"  {g}: " + ", ".join(f"{k}={v:.3f}" for k, v in report["wake"][g].items()))

    if args.replay:
        rep = load(args.replay)
        ref = min_delta(tab, "B", "B", "official")
        print("\n== cross-replay (E-fluid series from population S, agents not forcing)")
        print("   evolved = B-trained genomes, init = shared initial genomes")
        report["replay"] = {}
        for source in ("B", "init"):
            for v in ("official", "law"):
                al = {}
                for g in ("B", "init"):
                    al[g] = np.array([[r[v] for r in res["rows"]
                                       if r["genomes"] == g and r["series_from"] == source]
                                      for res in rep])
                d = (al["B"] - al["init"]).min(axis=1)
                res = paired(d, ref)
                key = f"series_from_{source}/{v}"
                report["replay"][key] = {
                    "median": float(np.median(d)), "evolved_mean": float(al["B"].mean()),
                    "init_mean": float(al["init"].mean()), **res}
                print(f"  series from {source:>4}, scored {v:>8}: evolved {al['B'].mean():.3f} "
                      f"init {al['init'].mean():.3f} median min_delta {np.median(d):.3f} "
                      f"change vs B {res['mean_diff']:+.3f} "
                      f"[{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] p={res['wilcoxon_p']:.4f} "
                      f"worse/better {res['worse']}/{res['better']}")
        own, foreign = {}, {}
        for g, src_own in (("B", "B"), ("init", "init")):
            for store, src in ((own, src_own), (foreign, "init" if g == "B" else "B")):
                store[g] = np.array([[r["official"] for r in res["rows"]
                                      if r["genomes"] == g and r["series_from"] == src]
                                     for res in rep]).mean(axis=1)
        print("\n  alignment of each genome set in its OWN wake vs a FOREIGN wake (official score)")
        for g in ("B", "init"):
            res = paired(own[g], foreign[g])
            report["replay"][f"own_vs_foreign/{g}"] = {
                "own": float(own[g].mean()), "foreign": float(foreign[g].mean()), **res}
            print(f"   {g:>4}: own {own[g].mean():.3f} foreign {foreign[g].mean():.3f} "
                  f"diff {res['mean_diff']:+.3f} [{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] "
                  f"p={res['wilcoxon_p']:.4f}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
