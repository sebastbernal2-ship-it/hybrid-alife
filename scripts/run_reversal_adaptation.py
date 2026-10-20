#!/usr/bin/env python
"""Bounded reversal-adaptation study.

The imposed law reverses at a declared step and the organism is not told.
Reports per-direction metrics for three conditions: adaptive internal estimate,
frozen estimate (memory-disabled control), and no reversal (control).

Run:
  JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py \
    --output docs/results/reversal_adaptation/summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_inference import train_estimate_policy_rwr
from hybrid_alife.experiments.latent_law_reversal import compare_reversal_adaptation

HORIZON = 16
REVERSAL_STEP = 8
DIRECTIONS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
SEEDS = (100, 101, 102, 103, 104)
ANGLES = (0, 22.5, 67.5, 90, 112.5, 157.5, 180, 202.5, 247.5, 270, 292.5, 337.5)
DENSE_TRAIN = tuple(
    (float(np.cos(np.deg2rad(a))), float(np.sin(np.deg2rad(a)))) for a in ANGLES
)


def _identity() -> dict[str, jnp.ndarray]:
    return {"w": jnp.eye(2), "b": jnp.zeros((2,))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-policy", action="store_true", default=True)
    args = parser.parse_args()

    identity = compare_reversal_adaptation(
        _identity(), DIRECTIONS, SEEDS, reversal_step=REVERSAL_STEP, horizon=HORIZON
    )

    trained_result = None
    if args.train_policy:
        policy, metadata = train_estimate_policy_rwr(
            seed=0,
            training_directions=DENSE_TRAIN,
            training_seeds=(0, 1, 2, 3),
            rounds=12,
            episodes_per_round=64,
            horizon=HORIZON,
        )
        trained_result = {
            "training": {
                "rounds": metadata["rounds"],
                "episodes_per_round": metadata["episodes_per_round"],
            },
            **compare_reversal_adaptation(
                policy,
                DIRECTIONS,
                SEEDS,
                reversal_step=REVERSAL_STEP,
                horizon=HORIZON,
            ),
        }

    artifact = {
        "schema_version": 1,
        "status": "calibration_only",
        "protocol": {
            "horizon": HORIZON,
            "reversal_step": REVERSAL_STEP,
            "transform": "reverse",
            "directions": [list(d) for d in DIRECTIONS],
            "seeds": list(SEEDS),
            "recovery": "alignment >= 0.8 for two consecutive post-reversal actions",
            "excluded_from_means": "step zero (no history) and the reversal step",
        },
        "identity_estimate_policy": identity,
        "trained_estimate_policy": trained_result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")

    print(f"artifact: {args.output}")
    for name, block in (
        ("identity", identity),
        ("trained", trained_result),
    ):
        if block is None:
            continue
        adaptive = block["adaptive"]
        frozen = block["frozen_estimate"]
        print(f"{name}: adaptive post-alignment min "
              f"{min(v['post_alignment'] for v in adaptive.values()):+.2f}, "
              f"frozen post-alignment max "
              f"{max(v['post_alignment'] for v in frozen.values()):+.2f}, "
              f"max recovery latency {block['gate']['max_recovery_latency']}")


if __name__ == "__main__":
    main()
