#!/usr/bin/env python
"""Formal gravity-study replication across seeds.

Runs the two positive architectures and the revelation treatments, writes one
JSON artifact, and prints a pass/fail gate per architecture.

Architectures:
  1. calibration end-to-end: a linear thrust policy learns the whole sensor to
     thrust mapping in the standalone calibration world.
  2. simulator internal estimate: the organism forms an explicit estimate of
     the law from proprioception, and a policy learns to act on it inside the
     real simulator world.

Held-out rule: training never sees the four 45 degree diagonal directions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.latent_law_continuous import (
    per_direction_continuous_rewards,
    train_continuous_policy_rwr,
)
from hybrid_alife.experiments.latent_law_inference import (
    per_direction_inference_rewards,
    run_inference_episode,
    train_estimate_policy_rwr,
)
from hybrid_alife.experiments.latent_law_task import GravityConfig

CARDINALS = ((0.0, -0.2), (0.0, 0.2), (-0.2, 0.0), (0.2, 0.0))
ANGLES = (0, 22.5, 67.5, 90, 112.5, 157.5, 180, 202.5, 247.5, 270, 292.5, 337.5)
DENSE = tuple(
    (float(np.cos(np.deg2rad(angle))), float(np.sin(np.deg2rad(angle))))
    for angle in ANGLES
)
DIAGONALS = (
    (0.14142, -0.14142),
    (0.14142, 0.14142),
    (-0.14142, 0.14142),
    (-0.14142, -0.14142),
)
UNIT_DIAGONALS = tuple(
    (float(x / np.hypot(x, y)), float(y / np.hypot(x, y)))
    for x, y in DIAGONALS
)
HORIZON = 12
GATE_WORST = 9.0
GATE_MEAN = 10.0


def _gate(scores: dict[str, float]) -> dict:
    values = list(scores.values())
    worst = float(min(values))
    mean = float(np.mean(values))
    return {
        "per_direction": {key: round(value, 4) for key, value in scores.items()},
        "worst": round(worst, 4),
        "mean": round(mean, 4),
        "passed": bool(worst > GATE_WORST and mean > GATE_MEAN),
    }


def calibration_end_to_end(seeds: tuple[int, ...]) -> list[dict]:
    results = []
    for seed in seeds:
        config = GravityConfig(horizon=HORIZON, gravity_choices=CARDINALS)
        policy, _ = train_continuous_policy_rwr(
            seed=seed,
            training_seeds=(0, 1, 2, 3),
            rounds=24,
            episodes_per_round=128,
            exploration_std=0.6,
            exploration_std_final=0.05,
            config=config,
        )
        scores = per_direction_continuous_rewards(
            policy, DIAGONALS, (100, 101, 102, 103), horizon=HORIZON
        )
        results.append({"seed": seed, **_gate(scores)})
    return results


def simulator_internal_estimate(seeds: tuple[int, ...]) -> list[dict]:
    results = []
    for seed in seeds:
        policy, _ = train_estimate_policy_rwr(
            seed=seed,
            training_directions=DENSE,
            training_seeds=(0, 1, 2, 3),
            rounds=12,
            episodes_per_round=64,
            horizon=HORIZON,
        )
        scores = per_direction_inference_rewards(
            policy, UNIT_DIAGONALS, (100, 101, 102, 103), horizon=HORIZON
        )
        results.append({"seed": seed, **_gate(scores)})
    return results


def revelations() -> dict[str, float]:
    import jax.numpy as jnp

    weights = jnp.asarray(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    policy = {"w": weights, "b": jnp.zeros((2,))}
    directions = ((1.0, 0.0), *UNIT_DIAGONALS)

    def mean_for(make) -> float:
        values = [
            run_inference_episode(
                policy,
                direction,
                horizon=HORIZON,
                use_revelation=True,
                revelation=make(direction),
            )["total_reward"]
            for direction in directions
        ]
        return float(np.mean(values))

    return {
        "no_revelation": round(mean_for(lambda _d: None), 4),
        "true_revelation": round(mean_for(lambda d: d), 4),
        "false_revelation": round(mean_for(lambda d: (-d[0], -d[1])), 4),
    }


def _summary(records: list[dict]) -> dict:
    worsts = [record["worst"] for record in records]
    means = [record["mean"] for record in records]
    passes = sum(1 for record in records if record["passed"])
    return {
        "seeds": len(records),
        "seeds_passed": passes,
        "worst_direction_mean": round(float(np.mean(worsts)), 4),
        "worst_direction_sd": round(float(np.std(worsts, ddof=1)), 4),
        "worst_single_direction": round(float(min(worsts)), 4),
        "mean_across_directions": round(float(np.mean(means)), 4),
        "mean_across_directions_sd": round(float(np.std(means, ddof=1)), 4),
        "gate": {"worst_above": GATE_WORST, "mean_above": GATE_MEAN},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seeds = tuple(args.seeds)

    calibration = calibration_end_to_end(seeds)
    simulator = simulator_internal_estimate(seeds)
    artifact = {
        "schema_version": 1,
        "status": "calibration_only",
        "protocol": {
            "horizon": HORIZON,
            "held_out": "four 45 degree diagonals, never trained on",
            "reward": "displacement: dot(unit_thrust, unit_law)",
            "learner": "reward-weighted regression, exploration annealed",
        },
        "calibration_end_to_end": {
            "per_seed": calibration,
            "summary": _summary(calibration),
        },
        "simulator_internal_estimate": {
            "per_seed": simulator,
            "summary": _summary(simulator),
        },
        "revelations": revelations(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"artifact: {args.output}")
    for name in ("calibration_end_to_end", "simulator_internal_estimate"):
        summary = artifact[name]["summary"]
        print(
            f"{name}: {summary['seeds_passed']}/{summary['seeds']} seeds pass, "
            f"worst-direction {summary['worst_direction_mean']:+.2f} "
            f"+/- {summary['worst_direction_sd']:.2f}, "
            f"mean {summary['mean_across_directions']:+.2f}"
        )
    revelations_result = artifact["revelations"]
    print(
        "revelations: "
        f"none {revelations_result['no_revelation']:+.2f}, "
        f"true {revelations_result['true_revelation']:+.2f}, "
        f"false {revelations_result['false_revelation']:+.2f}"
    )


if __name__ == "__main__":
    main()
