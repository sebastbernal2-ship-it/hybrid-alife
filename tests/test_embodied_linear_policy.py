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
from hybrid_alife.experiments.runner import initialize_sim


def _config():
    return embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=2,
        size=10,
        proprioception=True,
        linear_policy=True,
    )


def test_linear_policy_uses_a_flat_genome_and_runs() -> None:
    config = _config()
    state = initialize_sim(config)
    assert set(state.embodied.genomes) == {"w_linear", "b_linear"}
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    assert obs.shape[1] == embodied_observation_dim(config.embodied, config.world)
    actions, _ = act_embodied(state.embodied, obs, config.embodied, jax.random.PRNGKey(1))
    assert actions.shape == (4, embodied_action_dim(config.embodied))
    assert np.all(np.isfinite(np.asarray(actions)))


def test_linear_policy_can_express_the_flow_mapping() -> None:
    config = _config()
    state = initialize_sim(config)
    flow = jnp.array([[0.2, -0.1]], dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(flow[:, None, :], state.world.flow.shape)
    gain = config.embodied.flow_movement_scale
    pop = state.embodied
    pop.previous_velocity = jnp.broadcast_to(flow * gain, pop.previous_velocity.shape)
    pop.previous_thrust = jnp.zeros_like(pop.previous_thrust)
    obs = observe_embodied(
        pop, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    obs_dim = obs.shape[1]
    action_dim = embodied_action_dim(config.embodied)
    weights = jnp.zeros((pop.genomes["w_linear"].shape[0], obs_dim, action_dim))
    # Map the previous-velocity pair to the movement pair with a large gain.
    # A modest gain keeps tanh in its near-linear range, so the direction is
    # preserved. A very large gain saturates tanh and rotates the direction.
    weights = weights.at[:, -4, 0].set(10.0).at[:, -3, 1].set(10.0)
    pop.genomes = {
        "w_linear": weights,
        "b_linear": jnp.zeros((weights.shape[0], action_dim)),
    }
    actions, _ = act_embodied(pop, obs, config.embodied, jax.random.PRNGKey(2))
    unit = np.asarray(flow[0]) / np.linalg.norm(np.asarray(flow[0]))
    moves = np.asarray(actions[:, 0:2])
    alignment = (moves @ unit) / np.linalg.norm(moves, axis=-1)
    assert np.all(alignment > 0.99)


def test_linear_policy_evolution_pilot_runs() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=1,
        population=4,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        linear_policy=True,
    )
    assert result["status"] == "pilot_only"
    assert set(result["state"].embodied.genomes) == {"w_linear", "b_linear"}
    assert np.isfinite(result["training_alignment"]).all()
