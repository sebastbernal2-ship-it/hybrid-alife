"""Run a seeded field-discovery pilot in the hybrid-alife simulator."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, load_config, step_sim

SCHEMA_VERSION = 2
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


def run_evaluation(*, seed: int = 0, steps: int = 16, config_dir: str | Path = "configs") -> dict:
    """Run all four access conditions through the simulator's initialize/step runner."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    config_dir = Path(config_dir)
    conditions = {}
    for condition, config_name in _CONDITIONS.items():
        path = config_dir / config_name
        cfg = replace(load_config(path), seed=seed)
        if condition == "shuffled":
            cfg = replace(cfg, embodied=replace(cfg.embodied, shuffle_sixth_sense=True))
        state = initialize_sim(cfg)
        lineage_counter = cfg.embodied.population_size + (
            cfg.avida.population_size if cfg.avida.enabled else 0
        )
        split = steps // 2
        pre_config = cfg
        if condition == "intervention":
            pre_config = replace(cfg, world=replace(cfg.world, drifting=False))

        pre_metrics = None
        for step in range(steps):
            active_config = pre_config if step < split else cfg
            state, lineage_counter = step_sim(state, active_config, lineage_counter)
            if step + 1 == split:
                pre_metrics = _metrics(state)
        post_metrics = _metrics(state)
        if pre_metrics is None:
            raise RuntimeError("pilot did not capture its pre-intervention metrics")
        conditions[condition] = {
            "config": {
                "file": config_name,
                "seed": seed,
                "steps": steps,
                "intervention_step": split if condition == "intervention" else None,
                "hidden_field_groups": list(cfg.embodied.hidden_field_groups),
                "shuffle_sixth_sense": cfg.embodied.shuffle_sixth_sense,
            },
            "final_metrics": post_metrics,
            "pre_intervention": {"metrics": pre_metrics, "score": _score(pre_metrics)},
            "post_intervention": {"metrics": post_metrics, "score": _score(post_metrics)},
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "steps": steps,
        "conditions": conditions,
    }


def summarize_results(results: list[dict]) -> dict:
    """Summarize compact pilot artifacts across seeds."""
    if not results:
        raise ValueError("at least one field-discovery result is required")
    conditions = {}
    for result in results:
        if result.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported field-discovery schema version")
        if set(result.get("conditions", {})) != set(_CONDITIONS):
            raise ValueError("each result must contain all four conditions")
    for condition in _CONDITIONS:
        conditions[condition] = {}
        for metric in ("survival", "energy", "resource", "reproduction"):
            values = [
                float(run["conditions"][condition]["final_metrics"][metric])
                for run in results
            ]
            conditions[condition][metric] = {
                "mean": float(np.mean(values)),
                "n": len(values),
            }
        for phase in ("pre_intervention", "post_intervention"):
            values = [float(run["conditions"][condition][phase]["score"]) for run in results]
            conditions[condition][phase + "_score"] = {
                "mean": float(np.mean(values)),
                "n": len(values),
            }
    return {"schema_version": SCHEMA_VERSION, "run_count": len(results), "conditions": conditions}
