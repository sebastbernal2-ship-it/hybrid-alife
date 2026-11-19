"""Statistics for the rescored drag sweep (docs/results/embodied_law_lbm_live/drag_sweep_rescored).

Reads ``eval/seed*.json`` from ``measure_sweep_rescored.py`` and the committed per-seed sweep
JSONs, checks the reproduction gate, and prints the pre-specified tests.

Usage:
    python scripts/analyze_sweep_rescored.py --eval DIR/eval \
        --results docs/results/embodied_law_lbm_live --json OUT.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

DRAGS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
TARGETS = ("official", "law", "noself", "selfonly")
N_BOOT = 5000
REPRO = {
    0.0: "seeds0-14_B_live_nocoupling.json",
    0.5: "seeds0-14_C_mono_0.5.json",
    0.75: "drag_sweep/seeds0-14_monopole_0.75.json",
    1.0: "drag_sweep/seeds0-14_monopole_1.0.json",
    1.25: "drag_sweep/seeds0-14_monopole_1.25.json",
    1.5: "drag_sweep/seeds0-14_monopole_1.5.json",
    2.0: "seeds0-14_E_mono_2.0.json",
}


def holm(ps) -> np.ndarray:
    ps = np.asarray(ps, float)
    order = np.argsort(ps)
    adj = np.empty(len(ps))
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(ps) - rank) * ps[i])
        adj[i] = min(1.0, running)
    return adj


def per_seed_slope(drags, values) -> float:
    """Least-squares slope of one seed's min_delta against drag."""
    return float(np.polyfit(np.asarray(drags, float), np.asarray(values, float), 1)[0])


def boot_ci(x, rng, stat=np.mean) -> tuple[float, float]:
    x = np.asarray(x, float)
    draws = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def load_rows(eval_dir: Path) -> dict:
    """rows[(kind, drag, genomes, seed)] -> {direction: row}."""
    rows: dict = {}
    for f in sorted(Path(eval_dir).glob("seed*.json")):
        data = json.loads(f.read_text())
        for r in data["rows"]:
            key = (r["kind"], r["drag"], r["genomes"], data["seed"])
            rows.setdefault(key, {})[r["direction"]] = r
    return rows


def min_delta(rows, kind, drag, target, seed) -> float:
    """min over diagonals of (trained - initial), both scored in the same fluid and target."""
    trained = rows[(kind, drag, "trained", seed)]
    # Transfer rows are evaluated in the uncoupled fluid, whose initial reference is drag 0.
    init = rows[("own", drag if kind == "own" else 0.0, "init", seed)]
    return min(trained[d][target] - init[d][target] for d in trained)


def alignment(rows, kind, drag, genomes, target, seed) -> float:
    return float(np.mean([r[target] for r in rows[(kind, drag, genomes, seed)].values()]))


def analyse(rows, seeds, results: Path) -> dict:
    rng = np.random.default_rng(0)
    out: dict = {}
    # Reproduction gate against the committed per-seed numbers.
    gate = {}
    for drag, name in REPRO.items():
        ref = {r["seed"]: r["min_delta"] for r in json.loads((results / name).read_text())}
        gate[drag] = max(abs(min_delta(rows, "own", drag, "official", s) - ref[s]) for s in seeds)
    out["reproduction_max_abs_diff"] = {str(k): v for k, v in gate.items()}
    out["reproduction_ok"] = bool(max(gate.values()) < 1e-5)

    series = {}
    for label, kind, target in [(t, "own", t) for t in TARGETS] + [
        ("transfer", "transfer", "official")
    ]:
        series[label] = np.array(
            [
                [min_delta(rows, kind, d, target, s) for d in DRAGS if kind == "own" or d > 0]
                for s in seeds
            ]
        )
    out["per_drag"] = {}
    for label, arr in series.items():
        if label == "transfer":
            base = np.array([min_delta(rows, "own", 0.0, "official", s) for s in seeds])
            cols = arr
            drags = DRAGS[1:]
        else:
            base, cols, drags = arr[:, 0], arr[:, 1:], DRAGS[1:]
        ps, entries = [], {}
        for j, d in enumerate(drags):
            diff = cols[:, j] - base
            p = float(stats.wilcoxon(diff).pvalue) if np.any(diff != 0) else 1.0
            ps.append(p)
            lo, hi = boot_ci(diff, rng)
            entries[str(d)] = {
                "median": float(np.median(cols[:, j])),
                "mean_change": float(diff.mean()),
                "ci": [lo, hi],
                "worse": int((diff < 0).sum()),
                "better": int((diff > 0).sum()),
                "p": p,
            }
        for e, hp in zip(entries.values(), holm(ps), strict=True):
            e["holm_p"] = float(hp)
        out["per_drag"][label] = entries

    out["slopes"] = {}
    for label, arr in series.items():
        if label == "transfer":
            base = np.array([min_delta(rows, "own", 0.0, "official", s) for s in seeds])
            full = np.column_stack([base, arr])
        else:
            full = arr
        slopes = np.array([per_seed_slope(DRAGS, row) for row in full])
        lo, hi = boot_ci(slopes, rng)
        out["slopes"][label] = {
            "mean": float(slopes.mean()),
            "ci": [lo, hi],
            "negative": int((slopes < 0).sum()),
            "p_one_sided_negative": float(stats.wilcoxon(slopes, alternative="less").pvalue),
            "p_two_sided": float(stats.wilcoxon(slopes).pvalue),
        }
        out["slopes"][label]["per_seed"] = slopes.tolist()

    out["alignment"] = {}
    for drag in DRAGS:
        entry = {}
        for g in ("trained", "init"):
            for t in TARGETS:
                entry[f"{g}_{t}"] = float(
                    np.mean([alignment(rows, "own", drag, g, t, s) for s in seeds])
                )
        out["alignment"][str(drag)] = entry
    return out


def render(res: dict) -> str:
    lines = [
        f"reproduction ok: {res['reproduction_ok']}  max diffs: "
        + ", ".join(f"{k}: {v:.1e}" for k, v in res["reproduction_max_abs_diff"].items())
    ]
    lines.append("\n== per-seed slope of min_delta against drag (7 drags incl. 0)")
    for label, s in res["slopes"].items():
        lines.append(
            f"  {label:9s} mean {s['mean']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] "
            f"negative {s['negative']}/15  one-sided p = {s['p_one_sided_negative']:.4f}"
            f"  two-sided p = {s['p_two_sided']:.4f}"
        )
    lines.append(
        "\n== per drag: median min_delta, mean change vs drag 0 [95% CI], worse/better, p, Holm"
    )
    for label, entries in res["per_drag"].items():
        lines.append(f"  -- {label}")
        for d, e in entries.items():
            lines.append(
                f"    drag {d:>5s}: median {e['median']:.3f} change {e['mean_change']:+.3f} "
                f"[{e['ci'][0]:+.3f}, {e['ci'][1]:+.3f}] {e['worse']}/{e['better']} "
                f"p={e['p']:.4f} Holm={e['holm_p']:.4f}"
            )
    lines.append("\n== mean alignment by drag (own fluid; official / law / noself)")
    for d, a in res["alignment"].items():
        lines.append(
            f"  drag {d:>5s}: init {a['init_official']:.3f} / {a['init_law']:.3f} / "
            f"{a['init_noself']:.3f}   trained {a['trained_official']:.3f} / "
            f"{a['trained_law']:.3f} / {a['trained_noself']:.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", type=Path, required=True)
    ap.add_argument("--results", type=Path, default=Path("docs/results/embodied_law_lbm_live"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--txt", type=Path, default=None)
    args = ap.parse_args()
    rows = load_rows(args.eval)
    seeds = sorted({k[3] for k in rows})
    res = analyse(rows, seeds, args.results)
    text = render(res)
    print(text)
    if args.json:
        args.json.write_text(json.dumps(res, indent=1, sort_keys=True))
    if args.txt:
        args.txt.write_text(text + "\n")


if __name__ == "__main__":
    main()
