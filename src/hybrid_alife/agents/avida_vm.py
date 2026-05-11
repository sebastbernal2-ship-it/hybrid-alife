"""Avida-inspired instruction-genome virtual machine skeleton."""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp

from hybrid_alife.types import AvidaConfig, AvidaPopulationState, PRNGKey, WorldState


class Op(IntEnum):
    NOP = 0
    INC = 1
    DEC = 2
    ADD = 3
    NAND = 4
    SHIFT_L = 5
    SHIFT_R = 6
    LOAD_ENV = 7
    EMIT = 8
    H_ALLOC = 9
    H_COPY = 10
    H_DIVIDE = 11
    JUMP = 12
    JUMP_IF_ZERO = 13
    SET_READ = 14
    SET_WRITE = 15


INSTRUCTION_COUNT = len(Op)


def initialize_avida_population(cfg: AvidaConfig, key: PRNGKey) -> AvidaPopulationState:
    """Initialize digital organisms with random fixed-capacity genomes."""
    n = cfg.population_size
    genomes = jax.random.randint(key, (n, cfg.max_genome_length), 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    genome_lengths = jnp.full((n,), cfg.genome_length, dtype=jnp.int32)
    registers = jnp.zeros((n, cfg.registers), dtype=jnp.int32)
    memory = jnp.zeros((n, cfg.memory_size), dtype=jnp.int32)
    ip = jnp.zeros((n,), dtype=jnp.int32)
    read_head = jnp.zeros((n,), dtype=jnp.int32)
    write_head = jnp.zeros((n,), dtype=jnp.int32)
    copied = jnp.zeros((n, cfg.max_genome_length), dtype=jnp.int32)
    merit = jnp.ones((n,), dtype=jnp.float32)
    age = jnp.zeros((n,), dtype=jnp.int32)
    alive = jnp.ones((n,), dtype=bool)
    lineage_id = jnp.arange(n, dtype=jnp.int32)
    parent_id = -jnp.ones((n,), dtype=jnp.int32)
    return AvidaPopulationState(
        genomes=genomes,
        genome_lengths=genome_lengths,
        registers=registers,
        memory=memory,
        ip=ip,
        read_head=read_head,
        write_head=write_head,
        copied=copied,
        merit=merit,
        age=age,
        alive=alive,
        lineage_id=lineage_id,
        parent_id=parent_id,
    )


def step_avida_population(
    pop: AvidaPopulationState, world: WorldState, cfg: AvidaConfig, key: PRNGKey
) -> AvidaPopulationState:
    """Execute one update worth of VM cycles.

    V1 executes a fixed number of vectorized cycles. H_DIVIDE is staged as a TODO.
    """
    del key
    cycles = cfg.cycles_per_update
    for _ in range(cycles):
        pop = vm_cycle(pop, world, cfg)
    pop.age = pop.age + pop.alive.astype(jnp.int32)
    return pop


def vm_cycle(pop: AvidaPopulationState, world: WorldState, cfg: AvidaConfig) -> AvidaPopulationState:
    """Vectorized single-instruction dispatch skeleton."""
    idx = jnp.mod(pop.ip, pop.genome_lengths)
    op = jnp.take_along_axis(pop.genomes, idx[:, None], axis=1)[:, 0]

    reg0 = pop.registers[:, 0]
    reg1 = pop.registers[:, 1]
    reg0 = jnp.where(op == Op.INC, reg0 + 1, reg0)
    reg0 = jnp.where(op == Op.DEC, reg0 - 1, reg0)
    reg0 = jnp.where(op == Op.ADD, reg0 + reg1, reg0)
    reg0 = jnp.where(op == Op.NAND, jnp.bitwise_not(jnp.bitwise_and(reg0, reg1)), reg0)
    reg0 = jnp.where(op == Op.SHIFT_L, jnp.left_shift(reg0, 1), reg0)
    reg0 = jnp.where(op == Op.SHIFT_R, jnp.right_shift(reg0, 1), reg0)

    env_signal = jnp.asarray(jnp.mean(world.enrichment) * 1000.0, dtype=jnp.int32)
    reg0 = jnp.where(op == Op.LOAD_ENV, env_signal, reg0)

    pop.registers = pop.registers.at[:, 0].set(reg0)
    pop.ip = jnp.mod(pop.ip + 1, pop.genome_lengths)
    pop.merit = pop.merit + jnp.where(op == Op.EMIT, 0.01, 0.0)
    return pop


def mutate_avida_genomes(
    genomes: jax.Array, genome_lengths: jax.Array, cfg: AvidaConfig, key: PRNGKey
) -> tuple[jax.Array, jax.Array]:
    """Apply point mutations now; insertion/deletion is reserved for next implementation pass."""
    del genome_lengths
    k_mask, k_val = jax.random.split(key)
    mask = jax.random.bernoulli(k_mask, cfg.point_mutation_prob, genomes.shape)
    replacement = jax.random.randint(k_val, genomes.shape, 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    return jnp.where(mask, replacement, genomes), genome_lengths

