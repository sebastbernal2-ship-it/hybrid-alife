"""Transfer & robustness suite over ablations.

Fixed-policy transfer analysis: take a trained embodied checkpoint from
one environment and evaluate its frozen controller in another, then summarise
across pairings. This module is intentionally non-JIT and operates on
(config, checkpoint)
pairs so a single suite can be driven from a YAML matrix.

It also exposes `summarise_across_seeds(...)` returning median, IQR, and
Cliff's δ effect size — the headline reporting style required by the memo.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, step_sim
from hybrid_alife.metrics.core import collect_full_metrics
from hybrid_alife.replay.checkpoint import load_checkpoint, restore_sim_state
from hybrid_alife.types import ExperimentConfig, SimState

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


def _copy_frozen_policy(source: SimState, target: SimState) -> None:
    """Copy only embodied controller genomes from a source checkpoint."""
    if source.embodied is None or target.embodied is None:
        raise ValueError("fixed-policy transfer requires embodied agents in both runs")
    if set(source.embodied.genomes) != set(target.embodied.genomes):
        raise ValueError("source and target controller parameters do not match")
    for name, source_genome in source.embodied.genomes.items():
        target_genome = target.embodied.genomes[name]
        if source_genome.shape != target_genome.shape:
            raise ValueError(
                f"controller parameter {name!r} has incompatible shapes: "
                f"{source_genome.shape} != {target_genome.shape}"
            )
    target.embodied.genomes = {
        name: jnp.asarray(value) for name, value in source.embodied.genomes.items()
    }


def evaluate_fixed_policy(
    checkpoint_path: str | Path,
    target_cfg: ExperimentConfig,
    *,
    steps: int | None = None,
) -> dict[str, float]:
    """Evaluate a trained embodied policy in a fresh target world.

    The target starts with a new world and fresh agent state, then receives
    only the source controller genomes. Avida and reproduction are disabled,
    so no mutation, selection, or replacement birth can change the policy.
    """
    source_payload = load_checkpoint(checkpoint_path)
    source_state = restore_sim_state(source_payload)
    target_state = initialize_sim(target_cfg)
    _copy_frozen_policy(source_state, target_state)
    target_state.avida = None
    target_state.embodied_births_this_gen = 0
    target_state.embodied_deaths_this_gen = 0
    n_steps = steps
    if n_steps is None:
        n_steps = target_cfg.evolution.generations * target_cfg.evolution.steps_per_generation
    if n_steps < 0:
        raise ValueError("steps must be non-negative")
    lineage_counter = target_cfg.embodied.population_size
    for _ in range(n_steps):
        target_state, lineage_counter = step_sim(
            target_state,
            target_cfg,
            lineage_counter,
            allow_reproduction=False,
        )
    target_state.metrics = collect_full_metrics(target_state)
    return {key: float(value) for key, value in target_state.metrics.items()}


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
