#!/usr/bin/env python
"""Bounded revelation-form study.

Compares revelation channels and delivery timing against a no-revelation
baseline and a message-only control.

Run:
  JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py \
    --output docs/results/revelation_forms/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_inference import run_inference_episode

HORIZON = 12
EARLY = 0
LATE = 4
DIRECTIONS = (
    (0.0, -1.0),
    (1.0, 0.0),
    (0.7071, -0.7071),
    (0.7071, 0.7071),
    (-0.7071, 0.7071),
    (-0.7071, -0.7071),
)

# Thrust = estimate + revelation when present.
REVEAL = {
    "w": jnp.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ),
    "b": jnp.zeros((2,)),
}
# Thrust = revelation only; the estimate is ignored.
MESSAGE_ONLY = {
    "w": jnp.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    ),
    "b": jnp.zeros((2,)),
}


def _mean_reward(policy, *, revelation_maker, kind, step) -> float:
    values = []
    for direction in DIRECTIONS:
        result = run_inference_episode(
            policy,
            direction,
            horizon=HORIZON,
            use_revelation=True,
            revelation=revelation_maker(direction),
            revelation_step=step,
            revelation_kind=kind,
            revelation_noise=0.15 if kind == "noisy" else 0.0,
        )
        values.append(result["total_reward"])
    return float(np.mean(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    true_maker = lambda direction: direction  # noqa: E731
    false_maker = lambda direction: (-direction[0], -direction[1])  # noqa: E731
    none_maker = lambda direction: None  # noqa: E731

    conditions: dict[str, dict[str, float]] = {}
    for kind in ("exact", "demonstration", "noisy", "symbolic"):
        for timing, step in (("early", EARLY), ("late", LATE)):
            conditions[f"true_{kind}_{timing}"] = {
                "mean_reward": _mean_reward(
                    REVEAL, revelation_maker=true_maker, kind=kind, step=step
                )
            }
            conditions[f"false_{kind}_{timing}"] = {
                "mean_reward": _mean_reward(
                    REVEAL, revelation_maker=false_maker, kind=kind, step=step
                )
            }
    conditions["no_revelation"] = {
        "mean_reward": _mean_reward(
            REVEAL, revelation_maker=none_maker, kind="exact", step=EARLY
        )
    }
    conditions["message_only_true_exact_early"] = {
        "mean_reward": _mean_reward(
            MESSAGE_ONLY, revelation_maker=true_maker, kind="exact", step=EARLY
        )
    }
    conditions["message_only_false_exact_early"] = {
        "mean_reward": _mean_reward(
            MESSAGE_ONLY, revelation_maker=false_maker, kind="exact", step=EARLY
        )
    }

    baseline = conditions["no_revelation"]["mean_reward"]
    artifact = {
        "schema_version": 1,
        "status": "calibration_only",
        "protocol": {
            "horizon": HORIZON,
            "early_step": EARLY,
            "late_step": LATE,
            "noisy_sigma": 0.15,
            "symbolic": "nearest cardinal axis, lossy for diagonals",
            "directions": [list(d) for d in DIRECTIONS],
            "baseline_no_revelation": baseline,
        },
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"artifact: {args.output}")
    print(f"no revelation baseline: {baseline:+.2f}")
    for name in sorted(conditions):
        print(f"  {name:34s} {conditions[name]['mean_reward']:+.2f}")


if __name__ == "__main__":
    main()
