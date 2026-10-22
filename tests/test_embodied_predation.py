from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim


def _config(*, max_energy: float = 0.0, predation_rate: float = 0.0):
    base = embodied_law_config(
        seed=0,
        generations=1,
        population=2,
        steps_per_generation=1,
        size=10,
        proprioception=True,
    )
    return replace(
        base,
        embodied=replace(
            base.embodied,
            max_energy=max_energy,
            predation_rate=predation_rate,
            predation_radius=0.05,
        ),
    )


def test_energy_cap_limits_a_rich_organism() -> None:
    config = _config(max_energy=50.0)
    state = initialize_sim(config)
    pop = state.embodied
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.energy = jnp.array([100.0, 100.0])
    state.world.resources = jnp.ones_like(state.world.resources)
    actions = jnp.zeros((2, embodied_action_dim(config.embodied)))
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, config.embodied, config.world, jnp.array([0, 1])
    )
    assert float(pop.energy.max()) <= 50.0


def test_attacker_drains_a_nearby_target() -> None:
    config = _config(predation_rate=0.5)
    state = initialize_sim(config)
    pop = state.embodied
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.positions = jnp.array([[0.5, 0.5], [0.52, 0.5]], dtype=jnp.float32)
    pop.energy = jnp.array([20.0, 20.0])
    state.world.flow = jnp.zeros_like(state.world.flow)
    state.world.resources = jnp.zeros_like(state.world.resources)
    actions = jnp.zeros((2, embodied_action_dim(config.embodied)))
    actions = actions.at[0, 3].set(1.0)  # attacker attacks
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, config.embodied, config.world, jnp.array([0, 1])
    )
    assert float(pop.energy[0]) > 25.0
    assert float(pop.energy[1]) < 15.0
    assert bool(pop.last_attack[0])
    assert int(pop.last_attack_target[0]) == 1
