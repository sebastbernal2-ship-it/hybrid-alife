"""Analyses for the live-fluid and law-profile follow-ups, from committed per-seed JSONs only.

Prints: initial and evolved populations scored separately under the structured laws (seeds
0-14), the pre-registered 40-seed plain-channel test, the drag-2.0 lattice-speed maxima and
the masked-rerun check of the own-wake-removed target.

Usage: python scripts/analyze_lbm_followups.py [--root docs/results]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

PROFILE = {
    "uniform": "seeds0-14_uniform.json",
    "vectors-only": "seeds0-14_lbm.json",
    "rotated geometry": "seeds0-14_lbm_geometry.json",
    "plain channel": "seeds0-14_lbm_channel_geometry.json",
    "geometry + floor": "seeds0-14_lbm_geometry_flow_floor.json",
}


def load(path: Path) -> dict[int, dict]:
    return {r["seed"]: r for r in json.loads(path.read_text())}


def part(rows: dict, seeds, key: str) -> np.ndarray:
    """Per-seed mean over the four diagonals of ``evolved``, ``initial`` or ``reversed_law``."""
    return np.array([np.mean(list(rows[s][key].values())) for s in seeds])


def paired(diff: np.ndarray) -> dict:
    rng = np.random.default_rng(0)
    boot = diff[rng.integers(0, diff.size, (10000, diff.size))].mean(axis=1)
    return {
        "n": int(diff.size), "mean": float(diff.mean()),
        "ci95": [float(v) for v in np.percentile(boot, [2.5, 97.5])],
        "worse": int((diff < 0).sum()), "better": int((diff > 0).sum()),
        "wilcoxon_p": float(stats.wilcoxon(diff).pvalue) if np.any(diff) else 1.0,
    }


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    out, running = [0.0] * len(pvalues), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(pvalues) - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out


def fmt(r: dict) -> str:
    return (f"{r['mean']:+.3f} [{r['ci95'][0]:+.3f}, {r['ci95'][1]:+.3f}] "
            f"{r['worse']}/{r['better']} p={r['wilcoxon_p']:.4g}")


def sides(root: Path) -> None:
    data = {k: load(root / "embodied_law_lbm_profile" / v) for k, v in PROFILE.items()}
    seeds = range(15)
    print("Initial and evolved scored separately, seeds 0-14 (mean over diagonals)")
    for name, rows in data.items():
        print(f"  {name:18s} initial {part(rows, seeds, 'initial').mean():+.3f}  "
              f"evolved {part(rows, seeds, 'evolved').mean():+.3f}")
    names = [n for n in data if n != "uniform"]
    for key in ("initial", "evolved"):
        results = [paired(part(data[n], seeds, key) - part(data["uniform"], seeds, key))
                   for n in names]
        adjusted = holm([r["wilcoxon_p"] for r in results])
        for name, r, p in zip(names, results, adjusted, strict=True):
            print(f"  {key:8s} {name:18s} vs uniform: {fmt(r)} Holm(4)={p:.4g}")


def reversed_law_identity(root: Path) -> None:
    """The reversed-law run at diagonal d is the evolved run at -d (the diagonals are closed
    under negation and the rollout is deterministic), so min_reversed repeats min_evolved."""
    gaps = [
        abs(r["min_reversed"] - r["min_evolved"])
        for path in sorted(root.glob("embodied_law_lbm_*/**/seeds0-*.json"))
        for r in json.loads(path.read_text())
    ]
    print(f"\nmin_reversed vs min_evolved over {len(gaps)} rows: max |difference| {max(gaps):.2e}")


def extension(root: Path) -> None:
    base = root / "embodied_law_lbm_profile" / "extension40"
    uniform = load(base / "seeds0-39_uniform.json")
    channel = load(base / "seeds0-39_plain_channel.json")
    print("\n40-seed extension, plain channel minus uniform")
    for label, seeds in (("CONFIRMATORY 15-39", range(15, 40)), ("all 0-39", range(40)),
                         ("original 0-14", range(15))):
        diff = np.array([channel[s]["min_delta"] - uniform[s]["min_delta"] for s in seeds])
        a = np.array([uniform[s]["passed"] for s in seeds])
        b = np.array([channel[s]["passed"] for s in seeds])
        only_a, only_b = int((a & ~b).sum()), int((b & ~a).sum())
        fisher = stats.fisher_exact([[a.sum(), (~a).sum()], [b.sum(), (~b).sum()]]).pvalue
        mcnemar = (stats.binomtest(only_a, only_a + only_b).pvalue
                   if only_a + only_b else 1.0)
        print(f"  {label:20s} min_delta {fmt(paired(diff))}")
        print(f"  {'':20s} pass uniform {a.sum()}/{a.size} channel {b.sum()}/{b.size} "
              f"Fisher p={fisher:.4g} McNemar p={mcnemar:.4g}; median min_delta "
              f"{np.median([uniform[s]['min_delta'] for s in seeds]):.3f} / "
              f"{np.median([channel[s]['min_delta'] for s in seeds]):.3f}")
        for key in ("initial", "evolved"):
            d = part(channel, seeds, key) - part(uniform, seeds, key)
            print(f"  {'':20s} {key:12s} {fmt(paired(d))}")


def speeds(root: Path) -> None:
    rows = json.loads((root / "embodied_law_lbm_live" / "followups" / "speed.json").read_text())
    print("\nMax lattice speed over every sub-step at drag 2.0 (guard 0.3)")
    for cond in ("E", "F"):
        sel = sorted((r for r in rows if r["cond"] == cond), key=lambda r: r["seed"])
        gap = max(abs(r["min_delta"] - r["committed_min_delta"]) for r in sel)
        for phase in ("training", "evaluation", "end_of_step"):
            values = [r[phase] for r in sel]
            i = int(np.argmax(values))
            print(f"  {cond} {phase:11s} max {values[i]:.4f} (seed {sel[i]['seed']}) "
                  f"median {np.median(values):.4f}")
        print(f"  {cond} seeds {len(sel)}, max |min_delta - committed| {gap:.2e}")


def masked(root: Path) -> None:
    base = root / "embodied_law_lbm_live" / "followups" / "masked"
    seeds = [json.loads(p.read_text()) for p in sorted(base.glob("seed*.json"))]
    targets = ("official", "noself_superposition", "noself_open", "noself_closed")
    print(f"\nMasked reruns, fluid E, {len(seeds)} seeds")
    coupled = [r for s in seeds for r in s["rows"] if r["fluid"] == "E"]
    for name in ("B", "E", "init"):
        sel = [r for r in coupled if r["genomes"] == name]
        print(f"  {name:4s} alignment " + "  ".join(
            f"{t}={np.mean([r[t] for r in sel]):.3f}" for t in targets))
        print(f"  {'':4s} target error mean {np.mean([r['target_error_mean'] for r in sel]):.4f} "
              f"max {max(r['target_error_max'] for r in sel):.4f}; own wake "
              f"{np.mean([r['own_wake_mean'] for r in sel]):.4f}")

    def min_delta(seed: dict, genomes: str, fluid: str, target: str) -> float:
        def score(name, direction):
            for r in seed["rows"]:
                if (r["fluid"], r["genomes"], r["direction"]) == (fluid, name, direction):
                    return r["official" if fluid == "B" else target]
            raise KeyError((fluid, name, direction))
        directions = {r["direction"] for r in seed["rows"]}
        return min(score(genomes, d) - score("init", d) for d in directions)

    for name in ("B", "E"):
        for target in targets:
            diff = np.array([min_delta(s, name, "E", target) - min_delta(s, name, "B", target)
                             for s in seeds])
            print(f"  min_delta change E vs B, {name}-trained, {target:21s} mean "
                  f"{diff.mean():+.3f}  per seed {np.round(diff, 3).tolist()}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("docs/results"))
    root = parser.parse_args().root
    sides(root)
    reversed_law_identity(root)
    extension(root)
    speeds(root)
    masked(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
