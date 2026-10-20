from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_continuous import (
    per_direction_continuous_rewards,
    train_continuous_policy,
)
from hybrid_alife.experiments.latent_law_task import GravityConfig

CARDINALS = ((0.0, -0.2), (0.0, 0.2), (-0.2, 0.0), (0.2, 0.0))
DIAGONALS = (
    (0.14142, -0.14142),
    (0.14142, 0.14142),
    (-0.14142, 0.14142),
    (-0.14142, -0.14142),
)


def test_continuous_training_is_deterministic() -> None:
    config = GravityConfig(horizon=6, gravity_choices=CARDINALS)
    kwargs = {
        "seed": 3,
        "training_seeds": (0,),
        "updates": 2,
        "episodes_per_update": 4,
        "config": config,
    }
    first, first_meta = train_continuous_policy(**kwargs)
    second, second_meta = train_continuous_policy(**kwargs)
    assert first_meta == second_meta
    assert (first["w"] == second["w"]).all()


def test_plain_reinforce_collapses_to_a_fixed_direction() -> None:
    """Documents why reward-weighted regression is the adopted learner.

    Plain REINFORCE reaches a strong score on some directions and a strongly
    negative score on the opposite direction, which is the signature of a fixed
    thrust direction rather than gravity inference. Reward-weighted regression
    removes that failure.
    """
    config = GravityConfig(horizon=12, gravity_choices=CARDINALS)
    policy, _ = train_continuous_policy(
        seed=0,
        training_seeds=(0, 1, 2, 3),
        updates=300,
        episodes_per_update=32,
        config=config,
    )
    scores = per_direction_continuous_rewards(
        policy, DIAGONALS, (100, 101, 102, 103), horizon=12
    )
    values = list(scores.values())
    assert min(values) < 0.0, scores
    assert max(values) > 8.0, scores


def test_reward_weighted_regression_solves_every_held_out_direction() -> None:
    """The decisive gate: every held-out gravity direction must be solved.

    The policy is trained only on the four cardinal directions and deployed
    deterministically on four unseen diagonals. The hand-built optimum is +11
    per direction and every constant-action policy scores 0.0.
    """
    from hybrid_alife.experiments.latent_law_continuous import (
        train_continuous_policy_rwr,
    )

    config = GravityConfig(horizon=12, gravity_choices=CARDINALS)
    policy, metadata = train_continuous_policy_rwr(
        seed=0,
        training_seeds=(0, 1, 2, 3),
        rounds=24,
        episodes_per_round=128,
        exploration_std=0.6,
        exploration_std_final=0.05,
        config=config,
    )
    assert metadata["trainer"] == "reward_weighted_regression"
    scores = per_direction_continuous_rewards(
        policy, DIAGONALS, (100, 101, 102, 103), horizon=12
    )
    values = list(scores.values())
    assert min(values) > 8.0, scores
    assert float(np.mean(values)) > 9.0, scores




def test_linear_fit_unstandardises_correctly() -> None:
    """The fit must return coefficients for raw observations, not scaled ones."""
    from hybrid_alife.experiments.latent_law_continuous import fit_linear_thrust_policy

    rng = np.random.default_rng(0)
    observations = rng.normal(0.0, 2.0, size=(400, 4)) + np.asarray([5.0, -3.0, 0.5, 1.0])
    true_w = np.asarray([[1.0, 0.0], [0.0, 1.0], [2.0, -1.0], [-0.5, 0.5]])
    true_b = np.asarray([0.25, -0.75])
    targets = observations @ true_w + true_b
    policy = fit_linear_thrust_policy(observations, targets, np.ones(400), 1e-6)
    predicted = observations @ np.asarray(policy["w"]) + np.asarray(policy["b"])
    assert np.mean(np.abs(predicted - targets)) < 1e-4
