from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_inference import init_estimate_policy
from hybrid_alife.experiments.latent_law_reversal import (
    per_direction_reversal_metrics,
    run_reversal_episode,
)
from hybrid_alife.experiments.latent_law_simulator import SimulatorOrganism

IDENTITY = {"w": jnp.eye(2), "b": jnp.zeros((2,))}


def test_flow_reverses_at_the_declared_step_and_world_preserves_it() -> None:
    organism = SimulatorOrganism((0.0, -1.0), horizon=6, seed=0, reversal_step=2)
    assert np.allclose(organism.local_flow(), (0.0, -0.2))
    # The flip happens at the start of the action with index 2, the third call.
    for _ in range(3):
        organism.step(np.asarray((0.0, -1.0)))
    assert np.allclose(organism.local_flow(), (0.0, 0.2))
    # step_world must not overwrite the reversed law.
    assert np.allclose(np.asarray(organism.state.world.flow)[0, 0], (0.0, 0.2))


def test_estimate_policy_recovers_within_two_steps() -> None:
    result = run_reversal_episode(
        IDENTITY,
        (0.0, -1.0),
        reversal_step=8,
        horizon=16,
        transform="reverse",
    )
    assert result["pre_alignment"] > 0.9
    assert result["post_alignment"] > 0.8
    assert result["recovery_latency"] is not None
    assert result["recovery_latency"] <= 2, result["alignments"]


def test_recovery_latency_is_censored_for_a_fixed_direction_policy() -> None:
    # A policy that ignores the estimate and always thrusts down.
    fixed = {"w": jnp.zeros((2, 2)), "b": jnp.asarray([0.0, -1.0])}
    result = run_reversal_episode(
        fixed,
        (0.0, -1.0),
        reversal_step=8,
        horizon=16,
        transform="reverse",
    )
    assert result["pre_alignment"] > 0.9
    assert result["post_alignment"] < 0.0
    assert result["recovery_latency"] is None


def test_frozen_estimate_control_does_not_adapt() -> None:
    adaptive = run_reversal_episode(
        IDENTITY, (0.0, -1.0), reversal_step=8, horizon=16, transform="reverse"
    )
    frozen = run_reversal_episode(
        IDENTITY,
        (0.0, -1.0),
        reversal_step=8,
        horizon=16,
        transform="reverse",
        frozen_estimate=True,
    )
    assert adaptive["post_alignment"] > frozen["post_alignment"] + 1.5
    assert frozen["recovery_latency"] is None


def test_rotate_transform_breaks_the_antipodal_structure() -> None:
    result = run_reversal_episode(
        IDENTITY,
        (0.0, -1.0),
        reversal_step=8,
        horizon=16,
        transform="rotate",
    )
    assert result["pre_alignment"] > 0.9
    assert result["post_alignment"] > 0.8
    assert result["recovery_latency"] is not None


def test_metrics_are_reported_per_direction() -> None:
    metrics = per_direction_reversal_metrics(
        IDENTITY,
        ((0.0, -1.0), (1.0, 0.0)),
        (0, 1),
        reversal_step=6,
        horizon=12,
    )
    assert set(metrics) == {"(0.0, -1.0)", "(1.0, 0.0)"}
    for value in metrics.values():
        assert value["pre_alignment"] > 0.9
        assert value["recovered_all_seeds"] == 1


def test_estimate_policy_has_no_reversal_channel() -> None:
    """The reversal must not be disclosed: the policy input has no step field."""
    from hybrid_alife.experiments.latent_law_inference import encode_policy_input

    organism = SimulatorOrganism((0.0, -1.0), horizon=4, seed=0, reversal_step=1)
    organism.step(np.asarray((0.0, -1.0)))
    policy_input = encode_policy_input(organism.agent_observation, use_revelation=False)
    assert policy_input.shape == (2,)
    _ = init_estimate_policy(0)
