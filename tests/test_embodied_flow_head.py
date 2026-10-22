from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import (
    act_embodied,
    embodied_action_dim,
    observe_embodied,
)
from hybrid_alife.experiments.latent_law_embodied import (
    embodied_law_config,
    run_embodied_law_pilot,
)
from hybrid_alife.experiments.runner import initialize_sim, step_sim


def _config():
    return embodied_law_config(
        seed=0,
        generations=1,
        population=4,
        steps_per_generation=1,
        size=10,
        proprioception=True,
        unit_flow_features=True,
        flow_head=True,
        initial_alive_fraction=1.0,
    )


def test_flow_head_has_a_four_parameter_genome_and_runs() -> None:
    config = _config()
    state = initialize_sim(config)
    assert set(state.embodied.genomes) == {"w_flow", "b_flow"}
    assert state.embodied.genomes["w_flow"].shape == (4, 2, 2)
    assert state.embodied.genomes["b_flow"].shape == (4, 2)
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    actions, _ = act_embodied(state.embodied, obs, config.embodied, jax.random.PRNGKey(1))
    assert actions.shape == (4, embodied_action_dim(config.embodied))
    assert np.all(np.isfinite(np.asarray(actions)))


def test_flow_head_identity_aligns_with_the_flow() -> None:
    config = _config()
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


def test_flow_head_pilot_carries_the_flag() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=1,
        population=8,
        steps_per_generation=1,
        size=8,
        proprioception=True,
        unit_flow_features=True,
        flow_head=True,
        rank_replacement=True,
        immigrant_fraction=0.25,
    )
    assert result["status"] == "pilot_only"
    assert result["config"].embodied.flow_head is True
    assert set(result["state"].embodied.genomes) == {"w_flow", "b_flow"}


def test_flow_head_eat_gate_sets_the_eat_action() -> None:
    from dataclasses import replace

    import matplotlib  # noqa: F401  (kept for parity with the other head tests)

    config = _config()
    config = replace(
        config,
        embodied=replace(config.embodied, flow_head_eat_gate=0.9),
    )
    state = initialize_sim(config)
    obs = observe_embodied(
        state.embodied, state.world, config.embodied, config.world, jax.random.PRNGKey(0)
    )
    actions, _ = act_embodied(state.embodied, obs, config.embodied, jax.random.PRNGKey(1))
    import numpy as np

    assert np.allclose(np.asarray(actions[:, 2]), 0.9, atol=1e-3)
    assert np.all(np.asarray(actions[:, 3:8]) < 0.01)
