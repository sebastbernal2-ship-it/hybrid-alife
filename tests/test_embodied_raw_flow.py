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


def _config(*, raw_flow_features: bool, unit_flow_features: bool | None = None):
    if unit_flow_features is None:
        unit_flow_features = not raw_flow_features
    return embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        flow_head=True,
        unit_flow_features=unit_flow_features,
        raw_flow_features=raw_flow_features,
        initial_alive_fraction=1.0,
    )


def test_raw_flow_features_are_off_by_default() -> None:
    plain = _config(raw_flow_features=False, unit_flow_features=True)
    assert plain.embodied.raw_flow_features is False
    state = initialize_sim(plain)
    obs = observe_embodied(
        state.embodied, state.world, plain.embodied, plain.world, jax.random.PRNGKey(0)
    )
    assert obs.shape[1] == embodied_observation_dim(plain.embodied, plain.world)


def test_raw_flow_feature_is_the_raw_delta() -> None:
    config = _config(raw_flow_features=True, unit_flow_features=False)
    plain = embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        linear_policy=True,
    )
    assert embodied_observation_dim(config.embodied, config.world) == (
        embodied_observation_dim(plain.embodied, plain.world) + 2
    )
    state = initialize_sim(config)
    flow = jnp.array([[0.2, -0.1]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    state, _ = step_sim(state, config, 0, allow_reproduction=False)
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    expected = np.asarray(flow[0]) * config.embodied.flow_movement_scale
    assert np.allclose(np.asarray(obs[0, -2:]), expected, atol=1e-6)


def test_raw_flow_head_aligns_with_identity_weights() -> None:
    config = _config(raw_flow_features=True)
    state = initialize_sim(config)
    flow = jnp.array([[0.2, -0.1]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    state, _ = step_sim(state, config, 0, allow_reproduction=False)
    pop = state.embodied
    obs = observe_embodied(
        pop, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    identity = jnp.eye(2, dtype=jnp.float32)[None, :, :]
    pop.genomes = {
        "w_flow": jnp.broadcast_to(identity, (4, 2, 2)),
        "b_flow": jnp.zeros((4, 2)),
    }
    actions, _ = act_embodied(pop, obs, config.embodied, jax.random.PRNGKey(2))
    moves = np.asarray(actions[:, 0:2])
    unit = np.asarray(flow[0]) / np.linalg.norm(np.asarray(flow[0]))
    alignment = (moves @ unit) / np.linalg.norm(moves, axis=-1)
    assert np.all(alignment > 0.99)


def test_raw_flow_pilot_carries_the_flag() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=1,
        population=8,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        flow_head=True,
        raw_flow_features=True,
        rank_replacement=True,
        immigrant_fraction=0.25,
    )
    assert result["status"] == "pilot_only"
    assert result["config"].embodied.raw_flow_features is True
    assert actions_shape(result) == embodied_action_dim(result["config"].embodied)


def actions_shape(result) -> int:
    return int(result["state"].embodied.last_action.shape[1])
