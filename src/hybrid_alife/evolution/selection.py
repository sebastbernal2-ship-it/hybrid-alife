"""Selection, reseeding, and extinction handling."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.agents.embodied import (
    initialize_embodied_genomes,
    mutate_embodied_genomes,
)
from hybrid_alife.types import (
    AvidaPopulationState,
    EmbodiedConfig,
    EmbodiedPopulationState,
    EvolutionConfig,
    PRNGKey,
    WorldConfig,
)


def tournament_select(fitness: jax.Array, tournament_size: int, key: jax.Array) -> jax.Array:
    """Return one parent index per individual using vectorized tournament selection."""
    n = fitness.shape[0]
    candidates = jax.random.randint(key, (n, tournament_size), 0, n)
    candidate_fitness = fitness[candidates]
    winners = jnp.argmax(candidate_fitness, axis=1)
    return jnp.take_along_axis(candidates, winners[:, None], axis=1)[:, 0]


def reseed_embodied(
    pop: EmbodiedPopulationState,
    fitness: jax.Array,
    cfg: EmbodiedConfig,
    world_cfg: WorldConfig,
    key: PRNGKey,
) -> EmbodiedPopulationState:
    """If population is extinct or below threshold, reseed with random genomes.

    Otherwise leave pop unchanged; reproduction handles between-gen growth.
    """
    n = pop.alive.shape[0]
    alive_count = jnp.sum(pop.alive)

    # If everyone is dead, fully reset.
    # We can't branch on traced values cleanly; do it eagerly via .item()? It's a scalar.
    if int(alive_count) == 0:
        k_pos, k_genome = jax.random.split(key)
        genomes = initialize_embodied_genomes(cfg, world_cfg, k_genome)
        pop.genomes = genomes
        pop.positions = jax.random.uniform(k_pos, (n, 2), minval=0.0, maxval=1.0)
        pop.energy = jnp.full((n,), cfg.initial_energy, dtype=jnp.float32)
        pop.alive = jnp.ones((n,), dtype=bool)
        pop.age = jnp.zeros((n,), dtype=jnp.int32)
        pop.hidden = jnp.zeros_like(pop.hidden)
        pop.messages = jnp.zeros_like(pop.messages)
        if pop.previous_velocity is not None:
            pop.previous_velocity = jnp.zeros_like(pop.previous_velocity)
        if pop.previous_thrust is not None:
            pop.previous_thrust = jnp.zeros_like(pop.previous_thrust)
        pop.lineage_depth = jnp.zeros((n,), dtype=jnp.int32)
    _ = fitness
    return pop


def _rank_replace_embodied(
    pop: EmbodiedPopulationState,
    fitness: jax.Array,
    cfg: EmbodiedConfig,
    evo: EvolutionConfig,
    key: PRNGKey,
    world_cfg: WorldConfig | None = None,
) -> EmbodiedPopulationState:
    """Keep the top elite and replace the rest with mutated elite copies."""
    n = pop.alive.shape[0]
    n_elite = max(1, int(round(evo.elite_fraction * n)))
    n_immigrants = min(
        n - n_elite, max(0, int(round(cfg.immigrant_fraction * n)))
    )
    if n_immigrants > 0 and world_cfg is None:
        raise ValueError("immmigrant_fraction needs world_cfg")
    ranked = jnp.argsort(-jnp.asarray(fitness))
    elite_idx = ranked[:n_elite]
    immigrant_idx = ranked[n_elite : n_elite + n_immigrants]
    is_elite = jnp.zeros((n,), dtype=bool).at[elite_idx].set(True)
    is_immigrant = jnp.zeros((n,), dtype=bool).at[immigrant_idx].set(True)
    parent_idx = elite_idx[jnp.arange(n) % n_elite]
    k_fresh = jax.random.fold_in(key, 1)
    fresh = (
        initialize_embodied_genomes(cfg, world_cfg, k_fresh)
        if n_immigrants > 0
        else None
    )

    new_genomes: dict[str, jax.Array] = {}
    for genome_index, (name, value) in enumerate(pop.genomes.items()):
        parent_g = value[parent_idx]
        sub_key = jax.random.fold_in(key, genome_index)
        k_mask, k_noise = jax.random.split(sub_key)
        mask = jax.random.bernoulli(k_mask, cfg.genome_mutation_prob, parent_g.shape)
        noise = cfg.genome_mutation_std * jax.random.normal(k_noise, parent_g.shape)
        mutated = parent_g + mask * noise
        keep = is_elite.reshape((-1,) + (1,) * (value.ndim - 1))
        if fresh is not None:
            immigrant = is_immigrant.reshape((-1,) + (1,) * (value.ndim - 1))
            new_genomes[name] = jnp.where(
                keep, value, jnp.where(immigrant, fresh[name], mutated)
            )
        else:
            new_genomes[name] = jnp.where(keep, value, mutated)

    replaced = ~is_elite
    pop.genomes = new_genomes
    pop.energy = jnp.where(replaced, cfg.initial_energy * 0.5, pop.energy)
    pop.alive = jnp.ones((n,), dtype=bool)
    pop.age = jnp.where(replaced, 0, pop.age)
    pop.hidden = jnp.where(replaced[:, None], jnp.zeros_like(pop.hidden), pop.hidden)
    pop.messages = jnp.where(
        replaced[:, None], jnp.zeros_like(pop.messages), pop.messages
    )
    if pop.previous_velocity is not None:
        pop.previous_velocity = jnp.where(
            replaced[:, None], jnp.zeros_like(pop.previous_velocity), pop.previous_velocity
        )
    if pop.previous_thrust is not None:
        pop.previous_thrust = jnp.where(
            replaced[:, None], jnp.zeros_like(pop.previous_thrust), pop.previous_thrust
        )
    pop.lineage_depth = jnp.where(
        replaced, pop.lineage_depth[parent_idx] + 1, pop.lineage_depth
    )
    return pop


def select_and_mutate_embodied(
    pop: EmbodiedPopulationState,
    fitness: jax.Array,
    cfg: EmbodiedConfig,
    evo: EvolutionConfig,
    key: PRNGKey,
    *,
    world_cfg: WorldConfig | None = None,
) -> EmbodiedPopulationState:
    """End-of-generation selection: replace dead with mutated children from tournament parents."""
    if cfg.rank_replacement:
        return _rank_replace_embodied(pop, fitness, cfg, evo, key, world_cfg)
    k_sel, k_mut = jax.random.split(key)
    parent_idx = tournament_select(fitness, evo.tournament_size, k_sel)

    # Replace dead organisms with mutated parent copies.
    dead = ~pop.alive
    new_genomes: dict[str, jax.Array] = {}
    for genome_index, (name, value) in enumerate(pop.genomes.items()):
        parent_g = value[parent_idx]
        # Use the insertion-order index, not Python's process-randomized hash.
        # This keeps a seeded run reproducible across processes.
        sub_key = jax.random.fold_in(k_mut, genome_index)
        k_mask, k_noise = jax.random.split(sub_key)
        mask = jax.random.bernoulli(k_mask, cfg.genome_mutation_prob, parent_g.shape)
        noise = cfg.genome_mutation_std * jax.random.normal(k_noise, parent_g.shape)
        mutated = parent_g + mask * noise
        # broadcast dead mask
        dead_b = dead.reshape((-1,) + (1,) * (value.ndim - 1))
        if cfg.mutate_survivors:
            # Perturb every genome, including survivors. Dead slots still take the
            # mutated tournament parent; survivors keep their own genome plus noise.
            survivor_mutated = value + mask * noise
            new_genomes[name] = jnp.where(dead_b, mutated, survivor_mutated)
        else:
            new_genomes[name] = jnp.where(dead_b, mutated, value)
    _ = mutate_embodied_genomes  # imported for API completeness
    pop.genomes = new_genomes
    if pop.previous_velocity is not None:
        pop.previous_velocity = jnp.where(dead[:, None], 0.0, pop.previous_velocity)
    if pop.previous_thrust is not None:
        pop.previous_thrust = jnp.where(dead[:, None], 0.0, pop.previous_thrust)

    # Revive dead with starter state
    n = pop.alive.shape[0]
    pop.energy = jnp.where(dead, cfg.initial_energy * 0.5, pop.energy)
    pop.alive = jnp.where(dead, jnp.ones((n,), dtype=bool), pop.alive)
    pop.age = jnp.where(dead, 0, pop.age)
    pop.hidden = jnp.where(dead[:, None], jnp.zeros_like(pop.hidden), pop.hidden)
    pop.messages = jnp.where(dead[:, None], jnp.zeros_like(pop.messages), pop.messages)
    pop.lineage_depth = jnp.where(dead, pop.lineage_depth[parent_idx] + 1, pop.lineage_depth)
    pop.parent_id = jnp.where(dead, pop.lineage_id[parent_idx], pop.parent_id)
    return pop


def reseed_avida(pop: AvidaPopulationState, fraction: float, key: PRNGKey) -> AvidaPopulationState:
    """If too many digital organisms are dead, randomize their genomes to reseed."""
    n = pop.alive.shape[0]
    alive_count = jnp.sum(pop.alive)
    if int(alive_count) < max(1, int(fraction * n)):
        # full reseed
        from hybrid_alife.agents.avida_vm import INSTRUCTION_COUNT, Op

        max_len = pop.genomes.shape[1]
        new_random = jax.random.randint(
            key, pop.genomes.shape, 0, INSTRUCTION_COUNT, dtype=jnp.int32
        )
        template = jnp.array(
            [Op.H_ALLOC, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_DIVIDE],
            dtype=jnp.int32,
        )
        new_random = new_random.at[:, : template.shape[0]].set(
            jnp.broadcast_to(template, (n, template.shape[0]))
        )
        dead = ~pop.alive
        pop.genomes = jnp.where(dead[:, None], new_random, pop.genomes)
        pop.alive = jnp.ones((n,), dtype=bool)
        pop.ip = jnp.zeros((n,), dtype=jnp.int32)
        pop.read_head = jnp.zeros((n,), dtype=jnp.int32)
        pop.write_head = jnp.zeros((n,), dtype=jnp.int32)
        pop.copied = jnp.zeros((n, max_len), dtype=jnp.int32)
        pop.copied_length = jnp.zeros((n,), dtype=jnp.int32)
        pop.registers = jnp.zeros_like(pop.registers)
        pop.memory = jnp.zeros_like(pop.memory)
        pop.merit = jnp.ones((n,), dtype=jnp.float32)
        pop.age = jnp.zeros((n,), dtype=jnp.int32)
    return pop
