"""Reversal adaptation in the simulator gravity task.

The imposed law reverses at a declared step. The organism is not told. It can
only detect the change from its own sensors, so adaptation must come from its
internal estimate.

This module reports per-direction metrics, never an antipodal average:
pre-reversal alignment, post-reversal alignment, recovery latency with
censoring, and cumulative regret against a per-step oracle.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from hybrid_alife.experiments.latent_law_inference import (
    encode_policy_input,
    estimate_thrust,
    infer_law,
)
from hybrid_alife.experiments.latent_law_simulator import SimulatorOrganism
from hybrid_alife.types import WorldConfig

RECOVERY_ALIGNMENT = 0.8
RECOVERY_CONSECUTIVE = 2


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _rotate(vector: np.ndarray, degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.asarray(
        [c * vector[0] - s * vector[1], s * vector[0] + c * vector[1]],
        dtype=np.float64,
    )


def run_reversal_episode(
    policy: dict[str, Any],
    direction: tuple[float, float],
    *,
    reversal_step: int = 8,
    transform: str = "reverse",
    horizon: int = 16,
    seed: int = 0,
    config: WorldConfig | None = None,
    frozen_estimate: bool = False,
) -> dict[str, Any]:
    """Run one episode and report adaptation metrics.

    ``transform`` is ``reverse`` (law flips), ``rotate`` (law turns 90 degrees,
    which breaks the antipodal structure), or ``none`` (control).
    ``frozen_estimate`` keeps the pre-reversal estimate forever, which is the
    memory-disabled control.
    """
    if transform not in {"reverse", "rotate", "none"}:
        raise ValueError("transform must be 'reverse', 'rotate', or 'none'")
    if not 0 <= reversal_step < horizon:
        raise ValueError("reversal_step must be inside the episode")

    organism = SimulatorOrganism(direction, horizon=horizon, seed=seed, config=config)
    frozen: np.ndarray | None = None
    alignments: list[float] = []
    for step_index in range(horizon):
        observation = organism.agent_observation
        estimate = infer_law(observation)
        if step_index == reversal_step:
            if transform == "reverse":
                organism.state.world.flow = -organism.state.world.flow
            elif transform == "rotate":
                flow = np.asarray(organism.state.world.flow)
                rotated = _rotate(np.asarray(flow[0, 0]), 90.0)
                organism.state.world.flow = np.asarray(
                    np.broadcast_to(rotated, flow.shape), dtype=np.float32
                )
            if frozen_estimate:
                frozen = estimate
        if frozen is not None:
            estimate = frozen
        policy_input = encode_policy_input(observation, use_revelation=False)
        if frozen is not None:
            policy_input = estimate
        thrust = _unit(estimate_thrust(policy, policy_input))
        flow = organism.local_flow()
        flow_hat = _unit(flow)
        alignments.append(float(np.dot(thrust, flow_hat)))
        organism.step(thrust)

    # Step zero has no history, so its estimate is zero by construction, and the
    # reversal step's action was chosen before the flip. Neither measures
    # adaptation, so both are excluded from the alignment means. The regret
    # still counts the reversal step, because that cost is real.
    pre = alignments[1:reversal_step]
    post = alignments[reversal_step + 1 :]
    return {
        "direction": direction,
        "seed": seed,
        "transform": transform,
        "frozen_estimate": frozen_estimate,
        "alignments": alignments,
        "pre_alignment": float(np.mean(pre)) if pre else 0.0,
        "post_alignment": float(np.mean(post)) if post else 0.0,
        "recovery_latency": _recovery_latency(alignments, reversal_step),
        "regret": float(
            np.sum(1.0 - np.asarray(alignments[reversal_step:]))
        )
        if post
        else 0.0,
    }


def _recovery_latency(alignments: list[float], reversal_step: int) -> int | None:
    """Steps after reversal until alignment holds for consecutive actions."""
    values = alignments
    for start in range(reversal_step, len(values) - RECOVERY_CONSECUTIVE + 1):
        window = values[start : start + RECOVERY_CONSECUTIVE]
        if all(value >= RECOVERY_ALIGNMENT for value in window):
            return start - reversal_step
    return None


def per_direction_reversal_metrics(
    policy: dict[str, Any],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    reversal_step: int = 8,
    horizon: int = 16,
    transform: str = "reverse",
    frozen_estimate: bool = False,
    config: WorldConfig | None = None,
) -> dict[str, dict[str, float | int | None]]:
    """Per-direction metrics. Each direction is reported separately."""
    results: dict[str, dict[str, float | int | None]] = {}
    for direction in directions:
        records = [
            run_reversal_episode(
                policy,
                direction,
                reversal_step=reversal_step,
                transform=transform,
                horizon=horizon,
                seed=seed,
                config=config,
                frozen_estimate=frozen_estimate,
            )
            for seed in seeds
        ]
        latencies = [
            record["recovery_latency"]
            for record in records
            if record["recovery_latency"] is not None
        ]
        results[str(direction)] = {
            "pre_alignment": float(np.mean([r["pre_alignment"] for r in records])),
            "post_alignment": float(np.mean([r["post_alignment"] for r in records])),
            "recovered_all_seeds": int(len(latencies) == len(records)),
            "mean_recovery_latency": (
                float(np.mean(latencies)) if latencies else None
            ),
            "max_recovery_latency": (int(max(latencies)) if latencies else None),
            "regret": float(np.mean([r["regret"] for r in records])),
        }
    return results


def compare_reversal_adaptation(
    policy: dict[str, Any],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    reversal_step: int = 8,
    horizon: int = 16,
    config: WorldConfig | None = None,
) -> dict[str, Any]:
    """Adaptive estimate against the frozen-estimate and no-reversal controls."""
    adaptive = per_direction_reversal_metrics(
        policy,
        directions,
        seeds,
        reversal_step=reversal_step,
        horizon=horizon,
        transform="reverse",
        config=config,
    )
    frozen = per_direction_reversal_metrics(
        policy,
        directions,
        seeds,
        reversal_step=reversal_step,
        horizon=horizon,
        transform="reverse",
        frozen_estimate=True,
        config=config,
    )
    control = per_direction_reversal_metrics(
        policy,
        directions,
        seeds,
        reversal_step=reversal_step,
        horizon=horizon,
        transform="none",
        config=config,
    )
    return {
        "status": "calibration_only",
        "adaptive": adaptive,
        "frozen_estimate": frozen,
        "no_reversal": control,
        "gate": {
            "pre_alignment_min": min(v["pre_alignment"] for v in adaptive.values()),
            "post_alignment_min": min(v["post_alignment"] for v in adaptive.values()),
            "recovered_all_seeds": all(
                v["recovered_all_seeds"] for v in adaptive.values()
            ),
            "max_recovery_latency": max(
                (v["max_recovery_latency"] or horizon)
                for v in adaptive.values()
            ),
        },
    }


__all__ = [
    "RECOVERY_ALIGNMENT",
    "compare_reversal_adaptation",
    "per_direction_reversal_metrics",
    "run_reversal_episode",
]
