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
from typing import Any

import jax
import numpy as np

from hybrid_alife.experiments.latent_law_controller import (
    measure_law_detection,
    run_gravity_episode,
    train_gravity_actor_critic,
    train_gravity_controller,
)
from hybrid_alife.experiments.latent_law_task import (
    Action,
    GravityConfig,
    GravityEpisode,
    Treatment,
    scripted_oracle_action,
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
            action = scripted_oracle_action(episode)
        else:
            raise ValueError(f"unknown baseline: {policy}")
        rewards.append(episode.step(action).reward)
    return float(sum(rewards))


def _gravity_reward_and_alignment(
    params: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    horizon: int,
) -> dict[str, Any]:
    """Mean reward and gravity alignment on held-out gravity directions.

    Signed alignment cancels when the held-out directions are antipodal, so the
    headline number is the mean absolute alignment: how much of each step's
    movement lies on the gravity axis, in either direction. Rest gives 0 and a
    perfect diagonal-following policy gives 0.707 on these directions.
    """
    rewards: list[float] = []
    absolute: list[float] = []
    per_direction: dict[str, dict[str, float]] = {}
    for direction in directions:
        direction_rewards: list[float] = []
        direction_values: list[float] = []
        config = GravityConfig(horizon=horizon, gravity_choices=(direction,))
        for seed in seeds:
            result = run_gravity_episode(seed, Treatment.NO_REVELATION, params, config)
            rewards.append(result["total_reward"])
            direction_rewards.append(result["total_reward"])
            # Signed alignment is meaningful per direction. It must never be
            # averaged across antipodal directions, because that cancels it.
            signed = float(np.mean([i["gravity_alignment"] for i in result["evaluator_trace"]]))
            absolute.append(abs(signed))
            direction_values.append(signed)
        per_direction[str(direction)] = {
            "reward": float(np.mean(direction_rewards)),
            "signed_alignment": float(np.mean(direction_values)),
        }
    return {
        "reward": float(np.mean(rewards)) if rewards else 0.0,
        "alignment": float(np.mean(absolute)) if absolute else 0.0,
        "per_direction": per_direction,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--episodes-per-update", type=int, default=16)
    parser.add_argument("--training-seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--eval-seeds", type=int, nargs="+", default=[100, 101, 102, 103])
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--entropy-coef", type=float, default=0.05)
    parser.add_argument(
        "--trainer",
        choices=("policy_gradient", "actor_critic"),
        default="actor_critic",
    )
    parser.add_argument(
        "--include-no-benefit",
        action="store_true",
        help="Mix the no-benefit control into training instead of holding it out.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    train_config = GravityConfig(
        horizon=args.horizon,
        gravity_choices=_scaled(TRAIN_DIRECTIONS),
    )
    training_treatments = (Treatment.NO_REVELATION,)
    if args.include_no_benefit:
        training_treatments = (Treatment.NO_REVELATION, Treatment.NO_BENEFIT)
    if args.trainer == "actor_critic":
        trained, training = train_gravity_actor_critic(
            seed=args.seed,
            training_seeds=tuple(args.training_seeds),
            training_treatments=training_treatments,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            hidden=args.hidden,
            entropy_coef=args.entropy_coef,
            config=train_config,
        )
        params = trained["actor"]
    else:
        params, training = train_gravity_controller(
            seed=args.seed,
            training_seeds=tuple(args.training_seeds),
            training_treatments=training_treatments,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            hidden=args.hidden,
            config=train_config,
        )

    heldout = _scaled(HELDOUT_DIRECTIONS)
    learned = _gravity_reward_and_alignment(
        params, heldout, tuple(args.eval_seeds), args.horizon
    )
    detection = measure_law_detection(
        params, heldout, tuple(args.eval_seeds), horizon=args.horizon
    )
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
            "trainer": args.trainer,
            "include_no_benefit": args.include_no_benefit,
            "entropy_coef": args.entropy_coef,
            "updates": args.updates,
            "episodes_per_update": args.episodes_per_update,
            "hidden": args.hidden,
            "train_directions": _scaled(TRAIN_DIRECTIONS),
            "heldout_directions": heldout,
        },
        "training": training,
        "detection": detection,
        "heldout": learned,
        "heldout_gravity_alignment": learned["alignment"],
        "heldout_baselines": baselines,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"Gravity pilot written to {args.output}")
    print(
        f"learned held-out reward: {learned['reward']:.4f} "
        f"alignment: {learned['alignment']:.4f}"
    )
    print(
        f"law detection movement gap: {detection['movement_gap']:+.4f} "
        f"(present {detection['present_movement_rate']:.2f} "
        f"absent {detection['absent_movement_rate']:.2f})"
    )
    for policy, value in baselines.items():
        print(f"  baseline {policy}: {value:.4f}")


if __name__ == "__main__":
    main()
