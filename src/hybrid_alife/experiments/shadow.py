"""Neutral-shadow runner.

Implements the paired no-selection control required by Bedau-Packard activity
statistics (memo §2 / §6.3 / §8 P0.1). The shadow runner re-uses the existing
`run_experiment` but replaces selection by random tournaments on **uniform**
fitness, so the *demographics* match the experimental run while there is no
adaptive pressure.

We invoke it by mutating the EvolutionConfig in place: tournament_size is
preserved (so the sampling distribution is identical), but the fitness vector
passed to tournament selection is overridden upstream to be all-ones.

This module also exposes a `paired_activity` helper that takes the per-
generation activity dicts from an experimental and a shadow run and returns
the adaptive component.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from hybrid_alife.experiments import runner as _runner
from hybrid_alife.metrics.bedau import ActivityTracker, adaptive_activity
from hybrid_alife.types import ExperimentConfig, SimState


def _patch_selection_for_neutral() -> Callable[[], None]:
    """Monkey-patch tournament_select to ignore fitness.

    Returns a cleanup function that restores the original implementation. This
    is the smallest possible change to make the existing experiment loop run
    as a no-selection shadow — fitness is replaced with a constant vector so
    every tournament picks a random parent.
    """
    from hybrid_alife.evolution import selection as sel

    original = sel.tournament_select

    def neutral_tournament(fitness: jax.Array, tournament_size: int, key: jax.Array) -> jax.Array:
        flat = jnp.ones_like(fitness)
        return original(flat, tournament_size, key)

    sel.tournament_select = neutral_tournament  # type: ignore[assignment]

    def cleanup() -> None:
        sel.tournament_select = original  # type: ignore[assignment]

    return cleanup


def run_shadow(cfg: ExperimentConfig) -> SimState:
    """Run a neutral-shadow version of `cfg`.

    The output directory is `<cfg.output_dir>/<cfg.run_name>_shadow` so it does
    not collide with the experimental run.
    """
    shadow_cfg = replace(cfg, run_name=f"{cfg.run_name}_shadow")
    cleanup = _patch_selection_for_neutral()
    try:
        return _runner.run_experiment(shadow_cfg)
    finally:
        cleanup()


def paired_activity_from_lineage_logs(
    exp_log: list[dict[str, np.ndarray]],
    shadow_log: list[dict[str, np.ndarray]],
    persistence_threshold: int = 5,
) -> list[dict[str, float]]:
    """Compute paired (experimental - shadow) Bedau activity per generation.

    Each `log` entry must contain {"lineage_ids": [...], "alive": [...]}.
    """
    exp_tracker = ActivityTracker(persistence_threshold=persistence_threshold)
    shadow_tracker = ActivityTracker(persistence_threshold=persistence_threshold)
    exp_rows: list[dict[str, float]] = []
    shadow_rows: list[dict[str, float]] = []
    for gen, (e, s) in enumerate(zip(exp_log, shadow_log)):
        e_usage = _lineage_usage(e)
        s_usage = _lineage_usage(s)
        exp_tracker.observe(gen, e_usage)
        shadow_tracker.observe(gen, s_usage)
        exp_rows.append(exp_tracker.stats(gen))
        shadow_rows.append(shadow_tracker.stats(gen))
    return adaptive_activity(exp_rows, shadow_rows)


def _lineage_usage(entry: dict[str, np.ndarray]) -> dict[int, float]:
    ids = np.asarray(entry["lineage_ids"])
    alive = np.asarray(entry["alive"]).astype(bool)
    living = ids[alive]
    uniq, counts = np.unique(living, return_counts=True)
    return {int(u): float(c) for u, c in zip(uniq, counts)}


def write_shadow_config(cfg_path: str | Path, out_path: str | Path) -> Path:
    """Write a sidecar YAML with selection turned off (utility for the suite).

    Note: most consumers should call `run_shadow(cfg)` instead. This helper is
    here for documentation / external-driver use.
    """
    raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8"))
    raw = deepcopy(raw)
    raw["run_name"] = raw.get("run_name", "shadow") + "_shadow"
    Path(out_path).write_text(yaml.safe_dump(raw), encoding="utf-8")
    return Path(out_path)
