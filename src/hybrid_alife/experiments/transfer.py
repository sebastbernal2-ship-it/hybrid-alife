"""Transfer & robustness suite over ablations.

POET-style stepping-stone analysis (memo §3, P1.6): take a checkpoint from
one environment and evaluate it on another, then summarise across pairings.
This module is intentionally non-JIT and operates on (config, checkpoint)
pairs so a single suite can be driven from a YAML matrix.

It also exposes `summarise_across_seeds(...)` returning median, IQR, and
Cliff's δ effect size — the headline reporting style required by the memo.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------


def median_iqr(samples: np.ndarray) -> tuple[float, float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return 0.0, 0.0, 0.0
    q25, med, q75 = np.percentile(samples, [25, 50, 75])
    return float(med), float(q25), float(q75)


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta effect size in [-1, 1].

    +1 means every a > every b; 0 means full overlap; -1 means every a < b.
    The memo (§6.1) asks for an effect size *and* a CI on every headline
    difference; this is the simplest non-parametric choice.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    greater = np.sum(a[:, None] > b[None, :])
    less = np.sum(a[:, None] < b[None, :])
    return float((greater - less) / (a.size * b.size))


def bootstrap_ci(samples: np.ndarray, n_resample: int = 1000, seed: int = 0) -> tuple[float, float]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    medians = np.array(
        [np.median(rng.choice(samples, size=samples.size, replace=True)) for _ in range(n_resample)]
    )
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Suite construction
# ---------------------------------------------------------------------------


@dataclass
class AblationResult:
    name: str
    seed: int
    metrics: dict[str, float]


def aggregate(
    results: Iterable[AblationResult], metric: str
) -> dict[str, dict[str, float]]:
    """Group per-ablation results, return median/IQR/CI per metric."""
    by_name: dict[str, list[float]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(float(r.metrics.get(metric, 0.0)))
    out: dict[str, dict[str, float]] = {}
    for name, vals in by_name.items():
        arr = np.asarray(vals)
        med, q25, q75 = median_iqr(arr)
        lo, hi = bootstrap_ci(arr)
        out[name] = {
            "median": med,
            "q25": q25,
            "q75": q75,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": float(arr.size),
        }
    return out


def pairwise_effects(
    results: Iterable[AblationResult], metric: str, baseline: str
) -> dict[str, float]:
    """Cliff's δ of each ablation vs the named baseline on the given metric."""
    by_name: dict[str, list[float]] = {}
    for r in results:
        by_name.setdefault(r.name, []).append(float(r.metrics.get(metric, 0.0)))
    base = np.asarray(by_name.get(baseline, []))
    if base.size == 0:
        return {}
    return {
        name: cliffs_delta(np.asarray(vals), base)
        for name, vals in by_name.items()
        if name != baseline
    }


def write_summary_markdown(
    results: list[AblationResult],
    out_path: str | Path,
    metrics: list[str],
    baseline: str | None = None,
) -> None:
    """Write a transfer/robustness summary table to markdown."""
    lines = ["# Transfer / Robustness Suite — Summary\n"]
    for metric in metrics:
        lines.append(f"## {metric}\n")
        agg = aggregate(results, metric)
        lines.append("| Ablation | n | median | IQR | 95% CI |")
        lines.append("|---|---:|---:|---|---|")
        for name in sorted(agg):
            row = agg[name]
            lines.append(
                f"| {name} | {int(row['n'])} | {row['median']:.4f} | "
                f"[{row['q25']:.4f}, {row['q75']:.4f}] | "
                f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}] |"
            )
        if baseline is not None:
            effs = pairwise_effects(results, metric, baseline)
            if effs:
                lines.append(f"\n**Cliff's δ vs {baseline}:** "
                             + ", ".join(f"{k} = {v:+.3f}" for k, v in effs.items()))
        lines.append("")
    Path(out_path).write_text("\n".join(lines), encoding="utf-8")
