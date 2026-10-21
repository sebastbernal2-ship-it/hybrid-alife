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


def _population(*, rank_replacement: bool):
    base = embodied_law_config(
        seed=0, generations=1, population=8, steps_per_generation=1, size=10
    )
    config = replace(
        base,
        embodied=replace(
            base.embodied,
            rank_replacement=rank_replacement,
            genome_mutation_prob=1.0,
            genome_mutation_std=0.1,
        ),
    )
    state = initialize_sim(config)
    state.embodied.alive = jnp.ones_like(state.embodied.alive)
    # Distinct, known genomes per slot so a swap is visible.
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
        pop, fitness, config.embodied, config.evolution, jax.random.PRNGKey(0)
    )


def test_rank_replacement_preserves_the_best_genome() -> None:
    config, pop = _population(rank_replacement=True)
    before = {k: np.asarray(v) for k, v in pop.genomes.items()}
    out = _step(pop, config)
    best = 7  # highest fitness, top elite
    for name, value in out.genomes.items():
        assert np.allclose(np.asarray(value[best]), before[name][best]), name


def test_rank_replacement_replaces_the_worst_genome() -> None:
    config, pop = _population(rank_replacement=True)
    before = {k: np.asarray(v) for k, v in pop.genomes.items()}
    out = _step(pop, config)
    worst = 0  # lowest fitness, bottom replaced
    changed = [
        name
        for name, value in out.genomes.items()
        if not np.allclose(np.asarray(value[worst]), before[name][worst])
    ]
    assert changed, "expected the worst slot to be replaced"


def test_rank_replacement_resets_the_replaced_state() -> None:
    config, pop = _population(rank_replacement=True)
    pop.energy = jnp.zeros_like(pop.energy)
    pop.age = jnp.full_like(pop.age, 50)
    out = _step(pop, config)
    assert bool(out.alive[0])
    assert float(out.energy[0]) == config.embodied.initial_energy * 0.5
    assert int(out.age[0]) == 0


def test_rank_replacement_pilot_runs() -> None:
    result = run_embodied_law_pilot(
        seed=0,
        directions=((1.0, 0.0),),
        generations=2,
        population=4,
        steps_per_generation=1,
        size=8,
        rank_replacement=True,
    )
    assert result["status"] == "pilot_only"
    assert result["config"].embodied.rank_replacement is True
