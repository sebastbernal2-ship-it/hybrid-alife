"""Analyse the output of measure_flow_estimate_error.py.

Prints markdown tables: estimate error by condition, error-versus-margin association,
a mediation check for the monopole-versus-uncoupled margin gap, and the cross-evaluation
(train in one fluid, evaluate in another).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
from scipy import stats

DOC_FILES = {
    "A": "seeds0-14_A_legacy.json", "B": "seeds0-14_B_live_nocoupling.json",
    "C": "seeds0-14_C_mono_0.5.json", "D": "seeds0-14_D_dipole_0.5.json",
    "E": "seeds0-14_E_mono_2.0.json", "F": "seeds0-14_F_dipole_2.0.json",
}
LABEL = {"A": "legacy uniform", "B": "live, no coupling", "C": "monopole 0.5",
         "D": "dipole 0.5", "E": "monopole 2.0", "F": "dipole 2.0"}
SEEDS = list(range(15))
METRICS = ("angle_mean", "law_angle_mean", "magnitude_dev_mean", "frac_over_45")


def load(out: Path) -> dict[str, dict[str, np.ndarray]]:
    data = {}
    for cond in DOC_FILES:
        rows = [pickle.load(open(out / "cache" / f"{cond}_{s}.pkl", "rb"))["row"] for s in SEEDS]
        data[cond] = {k: np.array([r[k] for r in rows]) for k in ("min_delta", *METRICS)}
    return data


def wilcoxon(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    diff = a - b
    if np.allclose(diff, 0):
        return 0.0, 1.0
    return float(diff.mean()), float(stats.wilcoxon(diff).pvalue)


def ols_gap(y, is_e, x=None):
    cols = [np.ones_like(y), is_e] + ([x] if x is not None else [])
    beta, *_ = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)
    return beta


def mediation(d_b, d_e, mediator_b, mediator_e, rng, boots=4000):
    """Total, direct and indirect effect of condition E on min_delta through a mediator."""
    def effects(idx):
        y = np.concatenate([d_b[idx], d_e[idx]])
        m = np.concatenate([mediator_b[idx], mediator_e[idx]])
        flag = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))])
        total = ols_gap(y, flag)[1]
        beta = ols_gap(y, flag, m)
        return total, beta[1], total - beta[1], beta[2]

    point = effects(np.arange(len(d_b)))
    draws = np.array([effects(rng.integers(0, len(d_b), len(d_b))) for _ in range(boots)])
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    return point, lo, hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--docs", type=Path, default=Path("docs/results/embodied_law_lbm_live"))
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()
    data = load(args.out)
    rng = np.random.default_rng(0)
    report: dict = {}

    print("## Reproduction check against the recorded min_delta (max |difference| over 15 seeds)")
    for cond, name in DOC_FILES.items():
        rec = {r["seed"]: r["min_delta"] for r in json.load(open(args.docs / name))}
        diff = max(abs(rec[s] - data[cond]["min_delta"][i]) for i, s in enumerate(SEEDS))
        print(f"- {cond}: {diff:.2e}")
        report.setdefault("reproduction_max_abs_diff", {})[cond] = diff

    print("\n## Estimate error by condition (evolved genomes, 4 held-out diagonals, 15 seeds)")
    print("| Condition | angle mean (deg) | angle median over seeds | > 45 deg "
          "| flow-vs-law angle (deg) | flow magnitude dev | median min_delta "
          "| vs B: angle diff (Wilcoxon p) |")
    print("|---|---|---|---|---|---|---|---|")
    for cond in DOC_FILES:
        d = data[cond]
        mean_diff, p = wilcoxon(d["angle_mean"], data["B"]["angle_mean"])
        wins = int((d["angle_mean"] > data["B"]["angle_mean"]).sum())
        print(f"| {cond} {LABEL[cond]} | {d['angle_mean'].mean():.2f} | "
              f"{np.median(d['angle_mean']):.2f} | {d['frac_over_45'].mean():.3f} | "
              f"{d['law_angle_mean'].mean():.2f} | {d['magnitude_dev_mean'].mean():.3f} | "
              f"{np.median(d['min_delta']):.3f} | {mean_diff:+.2f} ({wins}/15 higher, p={p:.3g}) |")
        report.setdefault("by_condition", {})[cond] = {
            k: float(np.mean(d[k])) for k in ("min_delta", *METRICS)
        }

    print("\n## Does estimate error predict min_delta?")
    rows = []
    pooled = [c for c in DOC_FILES]
    live = ["C", "D", "E", "F"]
    for name, conds in (("all six conditions (n=90)", pooled), ("coupled C-F (n=60)", live)):
        x = np.concatenate([data[c]["angle_mean"] for c in conds])
        y = np.concatenate([data[c]["min_delta"] for c in conds])
        rho, p = stats.spearmanr(x, y)
        xr = np.concatenate([data[c]["angle_mean"] - data[c]["angle_mean"].mean() for c in conds])
        yr = np.concatenate([data[c]["min_delta"] - data[c]["min_delta"].mean() for c in conds])
        rho_w, p_w = stats.spearmanr(xr, yr)
        rows.append((name, rho, p, rho_w, p_w))
        report.setdefault("spearman", {})[name] = {"rho": rho, "p": p, "rho_within": rho_w,
                                                   "p_within": p_w}
    print("| Pool | Spearman rho (p) | rho after removing per-condition means (p) |")
    print("|---|---|---|")
    for name, rho, p, rho_w, p_w in rows:
        print(f"| {name} | {rho:+.2f} ({p:.3g}) | {rho_w:+.2f} ({p_w:.3g}) |")
    print("\n| Condition (n=15) | Spearman rho (p) |")
    print("|---|---|")
    for cond in live:
        rho, p = stats.spearmanr(data[cond]["angle_mean"], data[cond]["min_delta"])
        print(f"| {cond} {LABEL[cond]} | {rho:+.2f} ({p:.3g}) |")
        report.setdefault("spearman_per_condition", {})[cond] = {"rho": rho, "p": p}

    print("\n## Mediation check: how much of the min_delta gap against B runs through a mediator")
    print("Seed-resampled bootstrap (4000), 95% interval; pooled B + condition, OLS. total = gap,")
    print("direct = gap after adding the mediator, indirect = total - direct.\n")
    print("| Condition | Mediator | total gap | direct | indirect [95% CI] | mediator coef |")
    print("|---|---|---|---|---|---|")
    for cond in ("E", "F"):
        for mediator in ("angle_mean", "law_angle_mean", "magnitude_dev_mean"):
            (tot, dirc, ind, coef), lo, hi = mediation(
                data["B"]["min_delta"], data[cond]["min_delta"],
                data["B"][mediator], data[cond][mediator], rng)
            print(f"| {cond} | {mediator} | {tot:+.3f} | {dirc:+.3f} | {ind:+.3f} "
                  f"[{lo[2]:+.3f}, {hi[2]:+.3f}] | {coef:+.4f} |")
            report.setdefault("mediation", {})[f"{cond}_{mediator}"] = {
                "total": tot, "direct": dirc, "indirect": ind,
                "indirect_ci": [lo[2], hi[2]], "mediator_coef": coef}

    cross_dir = args.out / "cross"
    if cross_dir.exists():
        print("\n## Cross-evaluation: genomes trained in one fluid, evaluated in another")
        print("| Train -> eval | median min_delta | mean angle (deg) | vs same-fluid B->B: "
              "paired mean min_delta diff (Wilcoxon p, worse/15) |")
        print("|---|---|---|---|")
        base = data["B"]["min_delta"]
        table = {("B", "B"): data["B"], ("E", "E"): data["E"], ("F", "F"): data["F"]}
        for path in sorted(cross_dir.glob("train*_eval*_0.json")):
            tr, ev = path.name[5], path.name[11]
            rows = [json.load(open(cross_dir / f"train{tr}_eval{ev}_{s}.json")) for s in SEEDS]
            table[(tr, ev)] = {k: np.array([r[k] for r in rows]) for k in ("min_delta", *METRICS)}
            assert all(r["initial_genomes_identical_to_B"] for r in rows)
        for (tr, ev), d in table.items():
            mean_diff, p = wilcoxon(d["min_delta"], base)
            worse = int((d["min_delta"] < base).sum())
            print(f"| {tr} -> {ev} | {np.median(d['min_delta']):.3f} | {d['angle_mean'].mean():.2f}"
                  f" | {mean_diff:+.3f} (p={p:.3g}, {worse}/15) |")
            report.setdefault("cross", {})[f"{tr}->{ev}"] = {
                "median_min_delta": float(np.median(d["min_delta"])),
                "mean_diff_vs_BB": mean_diff, "p": p}
    if args.json:
        args.json.write_text(json.dumps(report, indent=1, default=float, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
