#!/usr/bin/env python
"""Run one bounded real-simulator causal-inquiry proxy bundle."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from hybrid_alife.experiments.causal_rollout import run_frozen_causal_rollout
from hybrid_alife.experiments.field_discovery import _condition_configs
from hybrid_alife.experiments.runner import load_config, run_experiment


def _causal_config(config):
    return replace(
        config,
        avida=replace(config.avida, enabled=False),
        world=replace(config.world, drifting=False),
    )


def run_bundle(*, seed: int, steps: int, training_generations: int, config_dir: Path) -> dict:
    if steps < 2:
        raise ValueError("steps must be at least two")
    if training_generations < 1:
        raise ValueError("training_generations must be positive")
    indirect = _causal_config(load_config(config_dir / "field_discovery_indirect.yaml"))
    source = replace(
        indirect,
        seed=seed,
        evolution=replace(indirect.evolution, generations=training_generations),
    )
    with TemporaryDirectory(prefix="causal-inquiry-source-") as output_dir:
        source = replace(source, output_dir=output_dir, run_name=f"causal-source-{seed}")
        trained = run_experiment(source)

    conditions = {
        "direct": _causal_config(load_config(config_dir / "field_discovery_direct.yaml")),
        "indirect": indirect,
        "decorrelated": _condition_configs(indirect, "decorrelated"),
        "no_field": _condition_configs(indirect, "no_field"),
        "intervention": _condition_configs(indirect, "intervention"),
    }
    reports = {}
    for name, config in conditions.items():
        reports[name] = run_frozen_causal_rollout(
            trained,
            replace(config, seed=seed),
            steps=steps,
            reverse_flow=name == "intervention",
        )
    reports["held_out_surface"] = run_frozen_causal_rollout(
        trained,
        indirect,
        steps=steps,
        flow_scale=1.5,
        evaluation_seed=seed + 1_000_003,
    )
    return {
        "schema_version": 1,
        "status": "proxy_calibration_only",
        "seed": seed,
        "steps": steps,
        "training_generations": training_generations,
        "conditions": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--training-generations", type=int, default=2)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_bundle(
        seed=args.seed,
        steps=args.steps,
        training_generations=args.training_generations,
        config_dir=args.config_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Causal rollout artifact written to {args.output}")


if __name__ == "__main__":
    main()
