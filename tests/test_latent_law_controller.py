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


def test_critic_learns_to_predict_returns() -> None:
    """A value head must reduce prediction error on a fixed batch."""
    import jax
    import jax.numpy as jnp

    from hybrid_alife.experiments.latent_law_controller import (
        OBS_DIM,
        critic_values,
        fit_critic_step,
        init_gravity_critic,
    )

    key = jax.random.PRNGKey(0)
    observations = jax.random.normal(key, (6, 4, OBS_DIM))
    targets = jnp.asarray([[1.0, 2.0, 3.0, 4.0]] * 6)
    critic = init_gravity_critic(seed=0, hidden=8)

    before = float(jnp.mean((critic_values(critic, observations) - targets) ** 2))
    for _ in range(400):
        critic = fit_critic_step(critic, observations, targets, learning_rate=0.05)
    after = float(jnp.mean((critic_values(critic, observations) - targets) ** 2))
    assert after < before * 0.5


def test_actor_critic_training_is_deterministic_and_declared() -> None:
    from hybrid_alife.experiments.latent_law_controller import (
        train_gravity_actor_critic,
    )
    from hybrid_alife.experiments.latent_law_task import GravityConfig

    kwargs = {
        "seed": 6,
        "training_seeds": (0,),
        "updates": 1,
        "episodes_per_update": 4,
        "config": GravityConfig(horizon=4),
    }
    first, first_meta = train_gravity_actor_critic(**kwargs)
    second, second_meta = train_gravity_actor_critic(**kwargs)
    assert first_meta == second_meta
    assert first_meta["trainer"] == "actor_critic"
    assert first_meta["entropy_coef"] >= 0
    for name in first["actor"]:
        assert (first["actor"][name] == second["actor"][name]).all()


def test_movement_rate_and_detection_gap_are_not_gameable_by_a_constant_policy() -> None:
    """A constant policy must score zero detection, an adaptive one above zero."""
    from hybrid_alife.experiments.latent_law_controller import (
        detection_gap,
        movement_rate,
    )

    def trace(actions):
        return {"agent_trace": [{"action": a} for a in actions]}

    moving = trace(["move_down"] * 10)
    resting = trace(["rest"] * 10)

    assert movement_rate(moving) == 1.0
    assert movement_rate(resting) == 0.0
    # A policy that always moves shows no discrimination.
    assert detection_gap([moving, moving], [moving, moving]) == 0.0
    # A policy that always rests also shows no discrimination.
    assert detection_gap([resting, resting], [resting, resting]) == 0.0
    # A policy that moves only when gravity is present discriminates fully.
    assert detection_gap([moving, resting], [resting, resting]) == 0.5


def test_target_reward_is_a_needle_landscape() -> None:
    """Documents a known defect in the target reward design.

    Riding gravity and then resting is worse than simply resting, for every
    partial ride. Riding and then braking is good at exactly one brake step,
    and its immediate neighbours are far worse. A reward with a single isolated
    optimum and no improvement path cannot be found by gradient policy search.
    The default reward is now "displacement", which removes the needle: see
    ``test_displacement_reward_has_no_needle``. This test pins the old design so
    the finding stays reproducible.
    """
    from hybrid_alife.experiments.latent_law_task import Action, GravityConfig, GravityEpisode

    config = GravityConfig(
        horizon=12,
        gravity_choices=((0.0, -0.2),),
        action_acceleration=0.40,
        reward_mode="target",
    )

    def total(ride_steps, later_action):
        episode = GravityEpisode(7, Treatment.NO_REVELATION, config)
        result = 0.0
        for step in range(12):
            action = Action.MOVE_DOWN if step < ride_steps else later_action
            result += episode.step(action).reward
        return result

    rest_only = total(0, Action.REST)
    ride_then_rest = [total(k, Action.REST) for k in range(1, 12)]
    # No improvement path: every partial ride is worse than doing nothing.
    assert all(value < rest_only for value in ride_then_rest), ride_then_rest

    ride_then_brake = [total(k, Action.MOVE_UP) for k in range(0, 12)]
    best = max(ride_then_brake)
    best_index = ride_then_brake.index(best)
    # One isolated optimum, with far worse neighbours on both sides.
    assert ride_then_brake[best_index - 1] < best - 10.0
    assert ride_then_brake[best_index + 1] < best - 10.0
    assert best > rest_only + 10.0


def test_linear_policy_training_is_deterministic() -> None:
    from hybrid_alife.experiments.latent_law_controller import (
        train_linear_gravity_policy,
    )
    from hybrid_alife.experiments.latent_law_task import GravityConfig

    config = GravityConfig(horizon=6, gravity_choices=((0.0, -0.2),), action_acceleration=0.40)
    kwargs = {
        "seed": 2,
        "training_seeds": (0,),
        "updates": 2,
        "episodes_per_update": 4,
        "config": config,
    }
    first, first_meta = train_linear_gravity_policy(**kwargs)
    second, second_meta = train_linear_gravity_policy(**kwargs)
    assert first_meta == second_meta
    assert (first["w"] == second["w"]).all()
