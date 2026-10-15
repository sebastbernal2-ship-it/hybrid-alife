#!/usr/bin/env python
"""Run the bounded learned-controller pilot for the explicit causal task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.causal_controller import (
    evaluate_causal_controller,
    train_causal_controller,
)
from hybrid_alife.experiments.causal_task import BenchmarkConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--training-generations", type=int, default=2)
    parser.add_argument("--population-size", type=int, default=4)
    parser.add_argument("--controller-hidden", type=int, default=8)
    parser.add_argument(
        "--objective",
        choices=("reward", "inquiry_shaped", "value_of_information"),
        default="reward",
        help="Training selector; inquiry_shaped uses evaluator-only shaping.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = BenchmarkConfig(horizon=args.horizon)
    params, training = train_causal_controller(
        seed=args.seed,
        training_seeds=tuple(args.training_seeds),
        generations=args.training_generations,
        population_size=args.population_size,
        hidden=args.controller_hidden,
        objective=args.objective,
        config=config,
    )
    evaluation = evaluate_causal_controller(
        params,
        seeds=tuple(args.seeds),
        config=config,
    )
    artifact = {
        "schema_version": 1,
        "status": "pilot_only",
        "config": {
            "horizon": args.horizon,
            "training_generations": args.training_generations,
            "population_size": args.population_size,
            "controller_hidden": args.controller_hidden,
            "objective": args.objective,
            "evaluation_seeds": args.seeds,
        },
        "training": training,
        "evaluation": evaluation,
        "parameters": {name: np.asarray(value).tolist() for name, value in params.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Causal controller pilot written to {args.output}")


if __name__ == "__main__":
    main()
