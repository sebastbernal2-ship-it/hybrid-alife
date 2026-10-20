from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.evolution.selection import select_and_mutate_embodied
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim


def _population(*, mutate_survivors: bool):
    base = embodied_law_config(
        seed=0, generations=1, population=8, steps_per_generation=1, size=10
    )
    config = replace(
        base,
        embodied=replace(
            base.embodied,
            mutate_survivors=mutate_survivors,
            genome_mutation_prob=1.0,
            genome_mutation_std=0.1,
        ),
    )
    state = initialize_sim(config)
    state.embodied.alive = jnp.ones_like(state.embodied.alive)
    return config, state.embodied


def _step(pop, config):
    fitness = jnp.arange(pop.alive.shape[0], dtype=jnp.float32)
    return select_and_mutate_embodied(
        pop, fitness, config.embodied, config.evolution, jax.random.PRNGKey(0)
    )


def test_survivors_are_frozen_by_default() -> None:
    config, pop = _population(mutate_survivors=False)
    before = {k: np.asarray(v) for k, v in pop.genomes.items()}
    out = _step(pop, config)
    for name, value in out.genomes.items():
        assert np.allclose(np.asarray(value), before[name]), name


def test_survivors_mutate_when_enabled() -> None:
    config, pop = _population(mutate_survivors=True)
    before = {k: np.asarray(v) for k, v in pop.genomes.items()}
    out = _step(pop, config)
    changed = [
        name
        for name, value in out.genomes.items()
        if not np.allclose(np.asarray(value), before[name])
    ]
    assert changed, "expected at least one survivor genome to change"


def test_survivor_mutation_is_a_small_perturbation() -> None:
    config, pop = _population(mutate_survivors=True)
    before = {k: np.asarray(v) for k, v in pop.genomes.items()}
    out = _step(pop, config)
    for name, value in out.genomes.items():
        delta = np.abs(np.asarray(value) - before[name])
        assert np.all(delta <= 1.0), f"{name} moved more than expected"
