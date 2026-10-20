#!/usr/bin/env python
"""Run the deterministic selective-causal-inquiry calibration benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.awareness_benchmark import (
    BenchmarkConfig,
    WorldKind,
    evaluate_benchmark,
    run_episode,
    train_controller,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--train-controller", action="store_true")
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--training-generations", type=int, default=8)
    parser.add_argument("--population-size", type=int, default=8)
    parser.add_argument("--controller-hidden", type=int, default=8)
    args = parser.parse_args()
    artifact = evaluate_benchmark(tuple(args.seeds), horizon=args.horizon)
    if args.train_controller:
        config = BenchmarkConfig(horizon=args.horizon)
        params, training = train_controller(
            seed=0,
            training_seeds=tuple(args.training_seeds),
            generations=args.training_generations,
            population_size=args.population_size,
            hidden=args.controller_hidden,
            config=config,
        )
        artifact["learned_controller"] = {
            "status": "controller_calibration_only",
            "training": training,
            "evaluation_seeds": list(args.seeds),
            "evaluation": {
                kind.value: [
                    run_episode(
                        kind,
                        seed,
                        "learned",
                        config,
                        controller_params=params,
                    )
                    for seed in args.seeds
                ]
                for kind in WorldKind
            },
            "parameters": {
                name: np.asarray(value).tolist() for name, value in params.items()
            },
        }
    payload = json.dumps(artifact, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Awareness benchmark written to {args.output}")


if __name__ == "__main__":
    main()
