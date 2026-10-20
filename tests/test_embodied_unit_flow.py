from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import (
    act_embodied,
    embodied_action_dim,
    embodied_observation_dim,
    observe_embodied,
)
from hybrid_alife.experiments.latent_law_embodied import (
    embodied_law_config,
    run_embodied_law_pilot,
)
from hybrid_alife.experiments.runner import initialize_sim, step_sim


def _config(*, unit_flow_features: bool):
    return embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        linear_policy=True,
        unit_flow_features=unit_flow_features,
        initial_alive_fraction=1.0,
    )


def _unit_flow_after_step(flow_magnitude: float):
    config = _config(unit_flow_features=True)
    state = initialize_sim(config)
    flow = jnp.array([[flow_magnitude, 0.0]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    state, _ = step_sim(state, config, 0, allow_reproduction=False)
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    return np.asarray(obs[0, -2:])


def test_unit_flow_features_are_off_by_default() -> None:
    plain = _config(unit_flow_features=False)
    assert plain.embodied.unit_flow_features is False
    state = initialize_sim(plain)
    obs = observe_embodied(
        state.embodied, state.world, plain.embodied, plain.world, jax.random.PRNGKey(0)
    )
    assert obs.shape[1] == embodied_observation_dim(plain.embodied, plain.world)


def test_unit_flow_feature_is_unit_length_and_scale_invariant() -> None:
    config = _config(unit_flow_features=True)
    plain = _config(unit_flow_features=False)
    assert embodied_observation_dim(config.embodied, config.world) == (
        embodied_observation_dim(plain.embodied, plain.world) + 2
    )
    small = _unit_flow_after_step(0.2)
    large = _unit_flow_after_step(0.4)
    assert np.isclose(np.linalg.norm(small), 1.0, atol=1e-4)
    assert np.allclose(small, large, atol=1e-4)
    assert np.allclose(small, [1.0, 0.0], atol=1e-4)


def test_linear_policy_aligns_with_order_one_weights() -> None:
    config = _config(unit_flow_features=True)
    state = initialize_sim(config)
    flow = jnp.array([[0.2, 0.0]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    state, _ = step_sim(state, config, 0, allow_reproduction=False)
    pop = state.embodied
    obs = observe_embodied(
        pop, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    obs_dim = obs.shape[1]
    action_dim = embodied_action_dim(config.embodied)
    weights = jnp.zeros((pop.genomes["w_linear"].shape[0], obs_dim, action_dim))
    # Order-one weights on the unit-flow feature are enough.
    weights = weights.at[:, -2, 0].set(1.0).at[:, -1, 1].set(1.0)
    pop.genomes = {
        "w_linear": weights,
        "b_linear": jnp.zeros((weights.shape[0], action_dim)),
    }
    actions, _ = act_embodied(pop, obs, config.embodied, jax.random.PRNGKey(2))
    moves = np.asarray(actions[:, 0:2])
    alignment = (moves @ np.array([1.0, 0.0])) / np.linalg.norm(moves, axis=-1)
    assert np.all(alignment > 0.99)


def test_unit_flow_evolution_pilot_runs() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=1,
        population=4,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        linear_policy=True,
        unit_flow_features=True,
    )
    assert result["status"] == "pilot_only"
    assert set(result["state"].embodied.genomes) == {"w_linear", "b_linear"}
