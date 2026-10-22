from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import (
    act_embodied,
    embodied_observation_dim,
    observe_embodied,
)
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim


def _config(*, food: bool, gain: float = 0.0):
    base = embodied_law_config(
        seed=0,
        generations=1,
        population=2,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        flow_head=True,
        unit_flow_features=True,
        food_gradient_features=food,
        food_gradient_gain=gain,
        initial_alive_fraction=1.0,
    )
    return base


def test_food_gradient_is_off_by_default_and_adds_two_features() -> None:
    plain = _config(food=False)
    with_food = _config(food=True)
    assert embodied_observation_dim(with_food.embodied, with_food.world) == (
        embodied_observation_dim(plain.embodied, plain.world) + 2
    )


def test_food_gradient_points_at_food() -> None:
    config = _config(food=True)
    state = initialize_sim(config)
    pop = state.embodied
    pop.positions = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32)
    resources = jnp.zeros_like(state.world.resources)
    resources = resources.at[5, 6, 0].set(1.0)
    state.world.resources = resources
    obs = observe_embodied(
        pop, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    grad = np.asarray(obs[0, -2:])
    assert grad[0] > 0.9
    assert abs(grad[1]) < 1e-6


def test_flow_head_food_reflex_moves_toward_food_without_a_law() -> None:
    config = _config(food=True, gain=1.0)
    state = initialize_sim(config)
    pop = state.embodied
    pop.positions = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32)
    resources = jnp.zeros_like(state.world.resources)
    resources = resources.at[5, 6, 0].set(1.0)
    state.world.resources = resources
    state.world.flow = jnp.zeros_like(state.world.flow)
    pop.genomes = {
        "w_flow": jnp.zeros((2, 2, 2)),
        "b_flow": jnp.zeros((2, 2)),
        "body_radius": jnp.ones((2,)),
        "body_aspect": jnp.ones((2,)),
    }
    obs = observe_embodied(
        pop, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    actions, _ = act_embodied(pop, obs, config.embodied, jax.random.PRNGKey(1))
    moves = np.asarray(actions[:, 0:2])
    assert np.all(moves[:, 0] > 0.5)
    assert np.all(np.abs(moves[:, 1]) < 0.01)
