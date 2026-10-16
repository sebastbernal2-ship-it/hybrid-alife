#!/usr/bin/env python
"""Bounded pilot: can an organism exploit a gravity law it was never told about?

Training exposes cardinal gravity directions only.
Evaluation uses held-out diagonal directions, so a constant-action policy
cannot succeed. A fixed "always move down" baseline is reported for contrast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import numpy as np

from hybrid_alife.experiments.latent_law_controller import (
    _ACTION_VECTORS,
    _ACTIONS,
    run_gravity_episode,
    train_gravity_controller,
)
from hybrid_alife.experiments.latent_law_task import (
    Action,
    GravityConfig,
    GravityEpisode,
    Treatment,
)

MAGNITUDE = 0.2
TRAIN_DIRECTIONS = ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0))
HELDOUT_DIRECTIONS = ((0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071))


def _scaled(directions: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    return tuple((x * MAGNITUDE, y * MAGNITUDE) for x, y in directions)


def _baseline_episode(
    policy: str,
    seed: int,
    treatment: Treatment,
    config: GravityConfig,
) -> float:
    episode = GravityEpisode(seed, treatment, config)
    rng = np.random.default_rng(seed + 4242)
    rewards: list[float] = []
    for _ in range(config.horizon):
        if policy == "rest":
            action = Action.REST
        elif policy == "move_down":
            action = Action.MOVE_DOWN
        elif policy == "random":
            action = Action(rng.choice(list(Action)))
        elif policy == "oracle":
            gravity = np.asarray(episode.gravity_vector)
            best = max(
                _ACTIONS,
                key=lambda candidate: float(
                    np.dot(_ACTION_VECTORS[candidate], gravity)
                    / max(float(np.linalg.norm(gravity)), 1e-8)
                ),
            )
            action = best
        else:
            raise ValueError(f"unknown baseline: {policy}")
        rewards.append(episode.step(action).reward)
    return float(sum(rewards))


def _heldout_alignment(
    params: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    horizon: int,
) -> float:
    """Mean movement alignment with the held-out gravity vector."""
    alignments: list[float] = []
    for direction in directions:
        config = GravityConfig(horizon=horizon, gravity_choices=(direction,))
        for seed in seeds:
            result = run_gravity_episode(seed, Treatment.NO_REVELATION, params, config)
            alignments.append(result["mean_gravity_alignment"])
    return float(np.mean(alignments)) if alignments else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_config = GravityConfig(
        horizon=args.horizon,
        gravity_choices=_scaled(TRAIN_DIRECTIONS),
    )
    params, training = train_gravity_controller(
        seed=args.seed,
        training_seeds=tuple(args.training_seeds),
        updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        hidden=args.hidden,
        config=train_config,
    )

    heldout = _scaled(HELDOUT_DIRECTIONS)
    learned = _heldout_alignment(params, heldout, tuple(args.eval_seeds), args.horizon)
    baselines: dict[str, float] = {}
    for policy in ("rest", "move_down", "random", "oracle"):
        values = [
            _baseline_episode(
                policy,
                seed,
                Treatment.NO_REVELATION,
                GravityConfig(horizon=args.horizon, gravity_choices=(direction,)),
            )
            for direction in heldout
            for seed in args.eval_seeds
        ]
        baselines[policy] = float(np.mean(values))

    artifact = {
        "schema_version": 1,
        "status": "pilot_only",
        "config": {
            "horizon": args.horizon,
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "hidden": args.hidden,
            "train_directions": _scaled(TRAIN_DIRECTIONS),
            "heldout_directions": heldout,
        },
        "training": training,
        "heldout_gravity_alignment": learned,
        "heldout_baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Gravity pilot written to {args.output}")
    print(f"learned held-out alignment: {learned:.4f}")
    for policy, value in baselines.items():
        print(f"  baseline {policy}: {value:.4f}")


if __name__ == "__main__":
    main()
