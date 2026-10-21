from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim


def _config(body_morphology: bool):
    base = embodied_law_config(
        seed=0,
        generations=1,
        population=2,
        steps_per_generation=1,
        size=10,
        proprioception=True,
    )
    return replace(
        base, embodied=replace(base.embodied, body_morphology=body_morphology)
    )


def test_body_genes_are_off_by_default_and_on_when_enabled() -> None:
    plain = _config(False)
    state = initialize_sim(plain)
    assert "body_radius" not in state.embodied.genomes
    assert "body_aspect" not in state.embodied.genomes

    config = _config(True)
    state = initialize_sim(config)
    assert state.embodied.genomes["body_radius"].shape == (2,)
    assert state.embodied.genomes["body_aspect"].shape == (2,)


def test_larger_body_gains_more_energy_from_food() -> None:
    config = _config(True)
    state = initialize_sim(config)
    pop = state.embodied
    pop.positions = jnp.array([[0.5, 0.5], [0.5, 0.5]], dtype=jnp.float32)
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.energy = jnp.full((2,), 20.0)
    state.world.resources = jnp.ones_like(state.world.resources)
    pop.genomes = {
        **pop.genomes,
        "body_radius": jnp.array([0.6, 1.8], dtype=jnp.float32),
        "body_aspect": jnp.ones((2,), dtype=jnp.float32),
    }
    actions = jnp.zeros((2, embodied_action_dim(config.embodied)), dtype=jnp.float32)
    actions = actions.at[:, 2].set(1.0)  # eat
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, config.embodied, config.world, jnp.array([0, 1])
    )
    assert float(pop.energy[1]) > float(pop.energy[0])
