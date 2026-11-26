"""Pre-registered analysis of the physics evolution study. See PREREGISTRATION.md."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy import stats

SEEDS = range(15)
BOOTSTRAP = 20_000
# Confirmatory family: one primary outcome per feature, each against the baseline.
PRIMARY = {
    "thermo": "emb_births",
    "signaling": "emit_rate_final_quarter",
    "stokes": "body_radius",
    "avida_energy": "av_genome_length",
    "lbm_dipole": "emb_births",
    "combined": "emb_lineage_depth",
}
# Exploratory only: unadjusted intervals, no confirmatory claims.
CONTROLS = (
    ("ledger", "baseline"),
    ("legacy_divide", "baseline"),
    ("lbm_oneway", "baseline"),
    ("lbm_dipole", "lbm_oneway"),
    ("avida_nocost", "baseline"),
)
OUTCOMES = (
    "emb_births", "emb_deaths", "av_births", "av_deaths", "emb_lineage_depth",
    "av_lineage_depth", "av_genome_length", "av_merit", "body_radius", "body_aspect",
    "emb_energy", "emit_rate_final_quarter", "emb_min_alive_end", "av_min_alive_end",
)


def sign_flip_p(diff: np.ndarray) -> float:
    """Exact two-sided paired sign-flip test of the mean difference."""
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(diff))))
    null = np.abs(signs @ diff)
    observed = abs(diff.sum())
    return float(np.mean(null >= observed - 1e-9 * max(1.0, observed)))


def holm(ps: list[float]) -> list[float]:
    order = np.argsort(ps)
    adjusted, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adjusted[i] = running
    return adjusted


def contrast(data: dict, arm: str, ref: str, outcome: str) -> dict:
    pairs = [(data[arm][s][outcome], data[ref][s][outcome]) for s in SEEDS]
    pairs = np.array([p for p in pairs if None not in p], dtype=float)
    if len(pairs) < 2:
        return {"arm": arm, "ref": ref, "outcome": outcome, "n": len(pairs)}
    diff = pairs[:, 0] - pairs[:, 1]
    rng = np.random.default_rng(12345)
    boot = diff[rng.integers(0, len(diff), (BOOTSTRAP, len(diff)))].mean(axis=1)
    sd = diff.std(ddof=1)
    return {
        "arm": arm, "ref": ref, "outcome": outcome, "n": len(diff),
        "arm_mean": float(pairs[:, 0].mean()), "ref_mean": float(pairs[:, 1].mean()),
        "mean_diff": float(diff.mean()), "sd_diff": float(sd),
        "dz": float(diff.mean() / sd) if sd > 0 else None,
        "ci95": [float(x) for x in np.percentile(boot, [2.5, 97.5])],
        "p": sign_flip_p(diff),
    }


def minimum_detectable_dz(n: int, alpha: float, power: float = 0.8) -> float:
    """Paired t approximation of the smallest standardised difference detectable."""
    critical = stats.t.ppf(1 - alpha / 2, n - 1)
    low, high = 0.0, 5.0
    for _ in range(60):
        mid = (low + high) / 2
        nc = mid * np.sqrt(n)
        achieved = stats.nct.sf(critical, n - 1, nc) + stats.nct.cdf(-critical, n - 1, nc)
        low, high = (mid, high) if achieved < power else (low, mid)
    return float(high)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="directory of <condition>/seed_<n>.json")
    args = parser.parse_args()
    arms = ["baseline", *PRIMARY, "ledger", "legacy_divide", "lbm_oneway", "avida_nocost"]
    data = {
        a: {s: json.loads((args.results / a / f"seed_{s}.json").read_text()) for s in SEEDS}
        for a in arms
    }
    primary = [contrast(data, arm, "baseline", out) for arm, out in PRIMARY.items()]
    for row, adjusted in zip(primary, holm([r.get("p", 1.0) for r in primary]), strict=True):
        row["p_holm"] = adjusted
    report = {
        "primary": primary,
        "controls_exploratory": [
            contrast(data, a, r, o) for a, r in CONTROLS for o in OUTCOMES
        ],
        "secondary_exploratory": [
            contrast(data, a, "baseline", o)
            for a, main_outcome in PRIMARY.items() for o in OUTCOMES if o != main_outcome
        ],
        "arm_summary": {
            a: {
                o: (float(np.mean(v)) if v else None)
                for o in OUTCOMES
                for v in [[data[a][s][o] for s in SEEDS if data[a][s][o] is not None]]
            }
            for a in arms
        },
        "minimum_detectable_dz": {
            "holm_worst_case_alpha": minimum_detectable_dz(15, 0.05 / len(PRIMARY)),
            "unadjusted": minimum_detectable_dz(15, 0.05),
        },
    }
    (args.results / "analysis.json").write_text(json.dumps(report, indent=1) + "\n")
    for row in primary:
        print(json.dumps(row))


if __name__ == "__main__":
    main()
