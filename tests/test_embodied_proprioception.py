from __future__ import annotations

import copy
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import (
    apply_embodied_actions,
    embodied_action_dim,
    embodied_observation_dim,
    observe_embodied,
)
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim, step_sim


def _config(*, proprioception: bool, displacement_reward: float = 0.0):
    base = embodied_law_config(
        seed=0, generations=1, population=8, steps_per_generation=1, size=10
    )
    return replace(
        base,
        embodied=replace(
            base.embodied,
            proprioception=proprioception,
            displacement_reward=displacement_reward,
        ),
    )


def test_proprioception_off_by_default_preserves_observation_shape() -> None:
    legacy = embodied_law_config(
        seed=0, generations=1, population=8, steps_per_generation=1, size=10
    )
    assert legacy.embodied.proprioception is False
    assert legacy.embodied.displacement_reward == 0.0

    state = initialize_sim(legacy)
    obs = observe_embodied(
        state.embodied, state.world, legacy.embodied, legacy.world, jax.random.PRNGKey(0)
    )
    assert obs.shape == (8, embodied_observation_dim(legacy.embodied, legacy.world))
    # The legacy proprio block is positions, energy, age, alive, hidden norm, messages.
    assert obs.shape[1] == embodied_observation_dim(legacy.embodied, legacy.world)
    assert state.embodied.previous_velocity is None
    assert state.embodied.previous_thrust is None


def test_proprioception_adds_four_features_and_recovers_the_law() -> None:
    config = _config(proprioception=True)
    state = initialize_sim(config)
    flow = jnp.array([[0.2, -0.1]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)

    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    legacy = _config(proprioception=False)
    assert obs.shape[1] == embodied_observation_dim(config.embodied, config.world)
    assert obs.shape[1] == embodied_observation_dim(legacy.embodied, legacy.world) + 4
    # The last four features are previous velocity then previous thrust.
    assert np.allclose(np.asarray(obs[0, -4:]), np.zeros(4), atol=1e-6)

    state, _ = step_sim(state, config, 0, allow_reproduction=False)
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    velocity = state.embodied.previous_velocity
    thrust = state.embodied.previous_thrust
    assert velocity is not None and thrust is not None
    expected_features = np.concatenate([velocity, thrust], axis=-1)
    assert np.allclose(np.asarray(obs[:, -4:]), expected_features, atol=1e-6)

    alive = np.asarray(state.embodied.alive)
    recovered = (
        np.asarray(velocity)[alive] - np.asarray(thrust)[alive]
    ) / config.embodied.flow_movement_scale
    expected = np.broadcast_to(np.asarray(flow[0]), recovered.shape)
    assert np.allclose(recovered, expected, atol=1e-4)


def test_displacement_reward_is_optional_and_directional() -> None:
    plain = _config(proprioception=False, displacement_reward=0.0)
    rewarded = _config(proprioception=False, displacement_reward=2.0)
    state = initialize_sim(plain)
    flow = jnp.array([[1.0, 0.0]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    state.embodied.energy = jnp.full_like(state.embodied.energy, 20.0)
    state.embodied.alive = jnp.ones_like(state.embodied.alive)

    actions = jnp.zeros((8, embodied_action_dim(plain.embodied)), dtype=jnp.float32)
    actions = actions.at[:, 0].set(1.0)  # push along +x, the law direction

    plain_pop = copy.deepcopy(state.embodied)
    rewarded_pop = copy.deepcopy(state.embodied)
    plain_pop, _, _ = apply_embodied_actions(
        plain_pop, actions, state.world, plain.embodied, plain.world, jax.random.PRNGKey(0)
    )
    rewarded_pop, _, _ = apply_embodied_actions(
        rewarded_pop, actions, state.world, rewarded.embodied, rewarded.world, jax.random.PRNGKey(0)
    )
    gain = np.asarray(rewarded_pop.energy) - np.asarray(plain_pop.energy)
    # Only the commanded +0.05 contributes; free advection is excluded.
    expected = 2.0 * 0.05
    assert np.allclose(gain, expected, atol=1e-4)
