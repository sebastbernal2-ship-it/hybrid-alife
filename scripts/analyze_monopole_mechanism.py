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
        print(f"  genomes {g} in fluid {fluid}: {np.median(min_delta(tab, fluid, g, 'official')):.3f}")

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
                report["min_delta"][f"{g}-trained/{fluid}/{v}"] = {"median": float(np.median(d)), **res}
                print(f"  {g}-trained in {fluid}, scored {v:>10}: median {np.median(d):6.3f} "
                      f"change {res['mean_diff']:+.3f} [{res['ci'][0]:+.3f},{res['ci'][1]:+.3f}] "
                      f"p={res['wilcoxon_p']:.4f} worse/better {res['worse']}/{res['better']}")

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
        print("\n== replay control (E-fluid series replayed, agents not forcing), B-trained")
        rep = load(args.replay)
        for scoring in ("full", "replay"):
            for v in ("official", "law"):
                ev, ini = [], []
                for res in rep:
                    for g, store in (("B", ev), ("init", ini)):
                        store.append([r[f"{scoring}_{v}"] for r in res["rows"] if r["genomes"] == g])
                d = (np.array(ev) - np.array(ini)).min(axis=1)
                ref = min_delta(tab, "B", "B", "official")
                res = paired(d, ref)
                report.setdefault("replay", {})[f"{scoring}/{v}"] = {"median": float(np.median(d)), **res}
                print(f"  {scoring:>6} rollout scored {v:>8}: median {np.median(d):6.3f} "
                      f"change vs B {res['mean_diff']:+.3f} p={res['wilcoxon_p']:.4f} "
                      f"worse/better {res['worse']}/{res['better']}")
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
