"""Simulator-backed field-discovery pilot and analysis helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, load_config, run_experiment, step_sim

SCHEMA_VERSION = 3
_CONDITIONS = {
    "direct": "field_discovery_direct.yaml",
    "indirect": "field_discovery_indirect.yaml",
    "shuffled": "field_discovery_direct.yaml",
    "intervention": "field_discovery_intervention.yaml",
}


def _metrics(state) -> dict[str, float | int]:
    embodied = state.embodied
    if embodied is None:
        alive = np.zeros(0, dtype=bool)
        energy = np.zeros(0, dtype=np.float32)
    else:
        alive = np.asarray(embodied.alive, dtype=bool)
        energy = np.asarray(embodied.energy, dtype=np.float64)
    return {
        "survival": float(alive.mean()) if alive.size else 0.0,
        "energy": float(energy[alive].mean()) if alive.any() else 0.0,
        "resource": float(np.asarray(state.world.resources, dtype=np.float64).sum()),
        "reproduction": int(state.embodied_births_this_gen),
    }


def _score(metrics: dict[str, float | int]) -> float:
    """Use survival-weighted mean energy as the pilot's functional score."""
    return float(metrics["survival"]) * float(metrics["energy"])


def _copy_trained_policy(trained_state, evaluation_state) -> None:
    if trained_state.embodied is None or evaluation_state.embodied is None:
        return
    evaluation_state.embodied.genomes = {
        name: jnp.array(value)
        for name, value in trained_state.embodied.genomes.items()
    }


def _evaluate_trajectory(
    trained_state,
    pre_cfg,
    post_cfg,
    *,
    steps: int,
) -> tuple[dict[str, float | int], dict[str, float | int]]:
    """Evaluate a frozen trained controller before and after one intervention."""
    state = initialize_sim(pre_cfg)
    _copy_trained_policy(trained_state, state)
    lineage_counter = pre_cfg.embodied.population_size + (
        pre_cfg.avida.population_size if pre_cfg.avida.enabled else 0
    )
    split = max(1, steps // 2)
    pre_metrics = None
    for step in range(steps):
        active_cfg = pre_cfg if step < split else post_cfg
        state, lineage_counter = step_sim(
            state,
            active_cfg,
            lineage_counter,
            allow_reproduction=False,
        )
        if step + 1 == split:
            pre_metrics = _metrics(state)
    if pre_metrics is None:
        raise RuntimeError("pilot did not capture pre-intervention metrics")
    return pre_metrics, _metrics(state)


def _condition_configs(cfg, condition: str):
    train_cfg = cfg
    if condition == "shuffled":
        train_cfg = replace(
            train_cfg,
            embodied=replace(train_cfg.embodied, shuffle_sixth_sense=True),
        )
    if condition == "intervention":
        train_cfg = replace(
            train_cfg,
            world=replace(train_cfg.world, drifting=False),
        )
    post_cfg = cfg if condition == "intervention" else train_cfg
    return train_cfg, post_cfg


def run_evaluation(
    *,
    seed: int = 0,
    steps: int = 16,
    training_generations: int = 2,
    config_dir: str | Path = "configs",
) -> dict[str, Any]:
    """Train each condition, then evaluate a frozen policy under an intervention."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    if training_generations < 1:
        raise ValueError("training_generations must be positive")
    config_dir = Path(config_dir)
    conditions = {}
    for condition, config_name in _CONDITIONS.items():
        cfg = replace(load_config(config_dir / config_name), seed=seed)
        train_cfg, post_cfg = _condition_configs(cfg, condition)
        evolution = replace(train_cfg.evolution, generations=training_generations)
        train_cfg = replace(train_cfg, evolution=evolution)
        with TemporaryDirectory(prefix=f"field-discovery-{condition}-") as output_dir:
            train_cfg = replace(
                train_cfg,
                output_dir=output_dir,
                run_name=f"train-{condition}-{seed}",
            )
            trained_state = run_experiment(train_cfg)
        pre_metrics, post_metrics = _evaluate_trajectory(
            trained_state,
            train_cfg,
            post_cfg,
            steps=steps,
        )
        conditions[condition] = {
            "config": {
                "file": config_name,
                "seed": seed,
                "steps": steps,
                "training_generations": training_generations,
                "intervention_step": steps // 2 if condition == "intervention" else None,
                "hidden_field_groups": list(train_cfg.embodied.hidden_field_groups),
                "shuffle_sixth_sense": train_cfg.embodied.shuffle_sixth_sense,
            },
            "final_metrics": post_metrics,
            "pre_intervention": {
                "metrics": pre_metrics,
                "score": _score(pre_metrics),
            },
            "post_intervention": {
                "metrics": post_metrics,
                "score": _score(post_metrics),
            },
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "steps": steps,
        "training_generations": training_generations,
        "conditions": conditions,
    }


def summarize_results(results: list[dict]) -> dict[str, Any]:
    """Summarize compact pilot artifacts across seeds."""
    if not results:
        raise ValueError("at least one field-discovery result is required")
    for result in results:
        if result.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported field-discovery schema version")
        if set(result.get("conditions", {})) != set(_CONDITIONS):
            raise ValueError("each result must contain all four conditions")
    conditions = {}
    for condition in _CONDITIONS:
        conditions[condition] = {}
        for metric in ("survival", "energy", "resource", "reproduction"):
            values = [
                float(run["conditions"][condition]["final_metrics"][metric])
                for run in results
            ]
            conditions[condition][metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
        for phase in ("pre_intervention", "post_intervention"):
            values = [
                float(run["conditions"][condition][phase]["score"])
                for run in results
            ]
            conditions[condition][phase + "_score"] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "n": len(values),
            }
    return {
        "schema_version": SCHEMA_VERSION,
        "run_count": len(results),
        "conditions": conditions,
    }
