from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.evolution.selection import select_and_mutate_embodied
from hybrid_alife.experiments.latent_law_embodied import (
    embodied_law_config,
    run_embodied_law_pilot,
)
from hybrid_alife.experiments.runner import initialize_sim


def _population(*, immigrant_fraction: float):
    base = embodied_law_config(
        seed=0, generations=1, population=8, steps_per_generation=1, size=10
    )
    config = replace(
        base,
        embodied=replace(
            base.embodied,
            rank_replacement=True,
            immigrant_fraction=immigrant_fraction,
            genome_mutation_prob=1.0,
            genome_mutation_std=0.1,
        ),
    )
    state = initialize_sim(config)
    state.embodied.alive = jnp.ones_like(state.embodied.alive)
    state.embodied.genomes = {
        name: jnp.broadcast_to(
            jnp.arange(8, dtype=jnp.float32).reshape((8,) + (1,) * (value.ndim - 1)),
            value.shape,
        )
        for name, value in state.embodied.genomes.items()
    }
    return config, state.embodied


def _step(pop, config):
    fitness = jnp.arange(pop.alive.shape[0], dtype=jnp.float32)
    return select_and_mutate_embodied(
        pop,
        fitness,
        config.embodied,
        config.evolution,
        jax.random.PRNGKey(0),
        world_cfg=config.world,
    )


def test_immigrants_are_fresh_and_the_middle_stays_near_the_elite() -> None:
    config, pop = _population(immigrant_fraction=0.25)
    # Elite slots get a large constant so fresh genomes are easy to tell apart.
    pop.genomes = {name: jnp.full_like(value, 1000.0) for name, value in pop.genomes.items()}
    out = _step(pop, config)
    # fitness 0..7, elite_fraction 0.5 -> top 4 = slots 4,5,6,7.
    for name, value in out.genomes.items():
        genome = np.asarray(value)
        assert np.allclose(genome[4:8], 1000.0), name
        # immigrant_fraction 0.25 -> 2 fresh slots = ranks 5,6 -> slots 3,2.
        assert np.all(np.abs(genome[2:4] - 1000.0) > 10.0), name
        # The remaining 2 replaced slots are mutated elite copies.
        assert np.all(np.abs(genome[0:2] - 1000.0) < 1.0), name


def test_immigrant_pilot_config_carries_the_flag() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=1,
        population=8,
        steps_per_generation=1,
        size=8,
        rank_replacement=True,
        immigrant_fraction=0.25,
    )
    assert result["status"] == "pilot_only"
    assert result["config"].embodied.immigrant_fraction == 0.25
