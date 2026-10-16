from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_controller import (
    evaluate_gravity_controller,
    run_gravity_episode,
    train_gravity_controller,
)
from hybrid_alife.experiments.latent_law_task import GravityConfig, Treatment


def test_gravity_controller_keeps_evaluator_labels_out_of_agent_trace() -> None:
    params, _ = train_gravity_controller(
        seed=2,
        training_seeds=(0,),
        updates=1,
        episodes_per_update=2,
        config=GravityConfig(horizon=4),
    )
    result = run_gravity_episode(
        seed=3,
        treatment=Treatment.NO_REVELATION,
        params=params,
        config=GravityConfig(horizon=4),
    )
    assert "gravity_vector" not in result["agent_trace"][0]["observation"]
    assert "gravity_vector" in result["evaluator_trace"][0]


def test_gravity_controller_training_is_deterministic() -> None:
    kwargs = {
        "seed": 4,
        "training_seeds": (0,),
        "updates": 1,
        "episodes_per_update": 2,
        "config": GravityConfig(horizon=4),
    }
    first, first_meta = train_gravity_controller(**kwargs)
    second, second_meta = train_gravity_controller(**kwargs)
    assert first_meta == second_meta
    for name in first:
        assert (first[name] == second[name]).all()


def test_gravity_controller_evaluates_revelation_and_no_benefit_conditions() -> None:
    params, _ = train_gravity_controller(
        seed=5,
        training_seeds=(0,),
        updates=1,
        episodes_per_update=2,
        config=GravityConfig(horizon=4),
    )
    result = evaluate_gravity_controller(
        params,
        seeds=(0,),
        config=GravityConfig(horizon=4),
    )
    assert set(result["conditions"]) == {t.value for t in Treatment}
    assert result["status"] == "calibration_only"


def test_training_loss_is_invariant_to_episode_order() -> None:
    """Episode boundaries must reset controller hidden state.

    If hidden state is carried across episodes, permuting episodes changes the
    loss. A correct per-episode rollout makes the order irrelevant.
    """
    import jax
    import jax.numpy as jnp

    from hybrid_alife.agents.controller import init_controller_params
    from hybrid_alife.experiments.latent_law_controller import (
        _ACTIONS,
        OBS_DIM,
        _policy_loss,
    )

    params = init_controller_params(1, OBS_DIM, len(_ACTIONS), 8, jax.random.PRNGKey(0))
    values = np.arange(2 * 3 * OBS_DIM, dtype="float32").reshape(2, 3, OBS_DIM) / 100.0
    observations = jnp.asarray(values)
    actions = jnp.asarray([[0, 1, 2], [3, 4, 0]])
    advantages = jnp.asarray([[1.0, -1.0, 0.5], [0.25, -0.5, 2.0]])

    forward_loss = _policy_loss(params, observations, actions, advantages)
    reversed_loss = _policy_loss(params, observations[::-1], actions[::-1], advantages[::-1])
    assert abs(float(forward_loss) - float(reversed_loss)) < 1e-6
