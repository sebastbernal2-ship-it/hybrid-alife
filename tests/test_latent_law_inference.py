from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_inference import (
    infer_law,
    init_estimate_policy,
    per_direction_inference_rewards,
    run_inference_episode,
    train_estimate_policy_rwr,
)
from hybrid_alife.experiments.latent_law_simulator import SimulatorOrganism

CARDINALS = ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0))
ANGLES = (0, 22.5, 67.5, 90, 112.5, 157.5, 180, 202.5, 247.5, 270, 292.5, 337.5)
DENSE = tuple(
    (float(np.cos(np.deg2rad(a))), float(np.sin(np.deg2rad(a)))) for a in ANGLES
)
DIAGONALS = (
    (0.7071, -0.7071),
    (0.7071, 0.7071),
    (-0.7071, 0.7071),
    (-0.7071, -0.7071),
)


def test_inferred_law_equals_the_imposed_law_after_one_step() -> None:
    """The internal estimate must equal the true law from proprioception."""
    organism = SimulatorOrganism((0.0, -1.0), horizon=4, seed=3)
    # Step zero has no history, so the estimate is zero by construction.
    assert np.allclose(infer_law(organism.agent_observation), (0.0, 0.0))
    organism.step(np.asarray((0.3, -0.9)))
    estimate = infer_law(organism.agent_observation)
    assert np.allclose(estimate, organism.local_flow(), atol=1e-9)
    assert np.allclose(estimate, (0.0, -0.2), atol=1e-9)


def test_identity_map_on_the_estimate_reaches_the_optimum() -> None:
    """If the policy trusts the estimate, it should score the maximum."""
    import jax.numpy as jnp

    policy = {"w": jnp.eye(2), "b": jnp.zeros((2,))}
    for direction in ((1.0, 0.0), (0.7071, -0.7071), (-0.7071, 0.7071)):
        result = run_inference_episode(policy, direction, horizon=12)
        # Step zero has no estimate, so the best achievable is 11 of 12.
        assert result["total_reward"] > 10.5, result["total_reward"]


def test_inference_policy_solves_every_held_out_direction() -> None:
    """The decisive gate for the internal-estimate architecture."""
    policy, metadata = train_estimate_policy_rwr(
        seed=0,
        training_directions=DENSE,
        training_seeds=(0, 1, 2, 3),
        rounds=12,
        episodes_per_round=64,
        horizon=12,
    )
    assert metadata["trainer"] == "internal_estimate_reward_weighted_regression"
    scores = per_direction_inference_rewards(
        policy, DIAGONALS, (100, 101, 102, 103), horizon=12
    )
    values = list(scores.values())
    assert min(values) > 9.0, scores
    assert float(np.mean(values)) > 10.0, scores


def test_no_estimate_control_cannot_solve_the_task() -> None:
    """Without the internal estimate the policy sees only a constant bias."""
    import jax.numpy as jnp

    # A policy that ignores the estimate is just a constant thrust direction.
    constant = {"w": jnp.zeros((2, 2)), "b": jnp.asarray([0.0, -1.0])}
    scores = per_direction_inference_rewards(
        constant, DIAGONALS, (100, 101, 102), horizon=12
    )
    values = list(scores.values())
    # A fixed direction is positive on some diagonals and negative on others.
    assert min(values) < 0.0, values
    assert max(values) > 0.0, values
    _ = init_estimate_policy(0)
