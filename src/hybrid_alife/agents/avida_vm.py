"""Avida-inspired instruction-genome virtual machine.

Vectorized over the population. Each digital organism has:
  - a genome (fixed-capacity int32 array of length max_genome_length)
  - an actual genome_length (logical length used)
  - registers, scratch memory, instruction pointer
  - read/write heads pointing into the parent and an offspring buffer
  - merit (proportional fitness from useful work)

Replication is the classic Avida pattern:
  H_ALLOC  : open offspring buffer of size max_genome_length
  H_COPY   : copy genome[read_head] -> copied[write_head], with mutation
  H_DIVIDE : if a chunk of contiguous content has been copied, spawn into a
             free slot, replacing a dead organism.

The execution loop is unrolled `cycles_per_update` times in Python; each
inner step is fully vectorized JAX.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp

from hybrid_alife.types import AvidaConfig, AvidaPopulationState, PRNGKey, WorldState
from hybrid_alife.world.env import positions_to_grid


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


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_avida_population(cfg: AvidaConfig, key: PRNGKey) -> AvidaPopulationState:
    """Initialize digital organisms.

    The first instructions of each genome are seeded with a viable self-replicator
    template (alloc + copy loop + divide) so that even at generation 0 some
    organisms can replicate, instead of needing many mutations to find replication.
    """
    n = cfg.population_size
    max_len = cfg.max_genome_length
    k_rand, k_seed = jax.random.split(key)
    genomes = jax.random.randint(k_rand, (n, max_len), 0, INSTRUCTION_COUNT, dtype=jnp.int32)

    # Seed a self-replicator at the start of each genome. The template is:
    #   H_ALLOC, H_COPY, H_COPY, H_COPY, H_COPY, H_DIVIDE
    # The VM auto-advances heads, so multiple H_COPY ops do progress.
    template = jnp.array(
        [Op.H_ALLOC, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_DIVIDE],
        dtype=jnp.int32,
    )
    # Mutate a per-organism amount: keep template intact in 80% of organisms.
    keep = jax.random.bernoulli(k_seed, 0.8, (n, template.shape[0]))
    seeded = jnp.where(keep, template[None, :], genomes[:, : template.shape[0]])
    genomes = genomes.at[:, : template.shape[0]].set(seeded)

    genome_lengths = jnp.full((n,), cfg.genome_length, dtype=jnp.int32)
    registers = jnp.zeros((n, cfg.registers), dtype=jnp.int32)
    memory = jnp.zeros((n, cfg.memory_size), dtype=jnp.int32)
    ip = jnp.zeros((n,), dtype=jnp.int32)
    read_head = jnp.zeros((n,), dtype=jnp.int32)
    write_head = jnp.zeros((n,), dtype=jnp.int32)
    copied = jnp.zeros((n, max_len), dtype=jnp.int32)
    copied_length = jnp.zeros((n,), dtype=jnp.int32)
    merit = jnp.ones((n,), dtype=jnp.float32)
    age = jnp.zeros((n,), dtype=jnp.int32)
    alive = jnp.ones((n,), dtype=bool)
    lineage_id = jnp.arange(n, dtype=jnp.int32)
    parent_id = -jnp.ones((n,), dtype=jnp.int32)
    lineage_depth = jnp.zeros((n,), dtype=jnp.int32)
    return AvidaPopulationState(
        genomes=genomes,
        genome_lengths=genome_lengths,
        registers=registers,
        memory=memory,
        ip=ip,
        read_head=read_head,
        write_head=write_head,
        copied=copied,
        copied_length=copied_length,
        merit=merit,
        age=age,
        alive=alive,
        lineage_id=lineage_id,
        parent_id=parent_id,
        lineage_depth=lineage_depth,
    )


# ---------------------------------------------------------------------------
# Per-cycle execution
# ---------------------------------------------------------------------------


def vm_cycle(
    pop: AvidaPopulationState,
    world: WorldState,
    cfg: AvidaConfig,
    positions: jax.Array | None,
    key: PRNGKey,
) -> AvidaPopulationState:
    """Execute one vectorized instruction per organism."""
    n = pop.genomes.shape[0]
    max_len = cfg.max_genome_length

    idx = jnp.mod(pop.ip, jnp.maximum(pop.genome_lengths, 1))
    op = jnp.take_along_axis(pop.genomes, idx[:, None], axis=1)[:, 0]

    reg = pop.registers
    reg0 = reg[:, 0]
    reg1 = reg[:, 1]
    reg2 = reg[:, 2] if cfg.registers >= 3 else jnp.zeros_like(reg0)

    # Arithmetic ops on reg0
    new_reg0 = reg0
    new_reg0 = jnp.where(op == Op.INC, new_reg0 + 1, new_reg0)
    new_reg0 = jnp.where(op == Op.DEC, new_reg0 - 1, new_reg0)
    new_reg0 = jnp.where(op == Op.ADD, reg0 + reg1, new_reg0)
    new_reg0 = jnp.where(op == Op.NAND, jnp.bitwise_not(jnp.bitwise_and(reg0, reg1)), new_reg0)
    new_reg0 = jnp.where(op == Op.SHIFT_L, jnp.left_shift(reg0, 1), new_reg0)
    new_reg0 = jnp.where(op == Op.SHIFT_R, jnp.right_shift(reg0, 1), new_reg0)

    # Environment IO
    if positions is not None and positions.shape[0] == n:
        h, w = world.enrichment.shape[0], world.enrichment.shape[1]
        grid = positions_to_grid(positions, h, w)
        env_per = (world.enrichment[grid[:, 0], grid[:, 1], 0] * 1000.0).astype(jnp.int32)
        conc_per = (world.concentration[grid[:, 0], grid[:, 1], 0] * 1000.0).astype(jnp.int32)
    else:
        env_per = jnp.full((n,), int(jnp.mean(world.enrichment) * 1000.0), dtype=jnp.int32)
        conc_per = jnp.full((n,), int(jnp.mean(world.concentration) * 1000.0), dtype=jnp.int32)

    new_reg0 = jnp.where(op == Op.LOAD_ENV, env_per, new_reg0)
    new_reg1 = jnp.where(op == Op.LOAD_ENV, conc_per, reg1)
    # SET_READ/SET_WRITE: put register 0 mod max_len into the corresponding head
    new_read = jnp.where(op == Op.SET_READ, jnp.mod(reg0, max_len), pop.read_head)
    new_write = jnp.where(op == Op.SET_WRITE, jnp.mod(reg0, max_len), pop.write_head)

    # H_ALLOC: reset copied buffer length to 0, write head to 0
    is_alloc = op == Op.H_ALLOC
    new_copied_length = jnp.where(is_alloc, 0, pop.copied_length)
    new_write = jnp.where(is_alloc, 0, new_write)
    new_read_after_alloc = jnp.where(is_alloc, 0, new_read)

    # H_COPY: copy genomes[read_head] -> copied[write_head], advance both heads
    is_copy = op == Op.H_COPY
    src_idx = jnp.mod(new_read_after_alloc, jnp.maximum(pop.genome_lengths, 1))
    src_val = jnp.take_along_axis(pop.genomes, src_idx[:, None], axis=1)[:, 0]
    # apply point mutation during copy
    key, k_mut = jax.random.split(key)
    mut_mask = jax.random.bernoulli(k_mut, cfg.point_mutation_prob, (n,))
    key, k_val = jax.random.split(key)
    mut_val = jax.random.randint(k_val, (n,), 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    copy_val = jnp.where(mut_mask, mut_val, src_val)
    write_idx = jnp.mod(new_write, max_len)
    copied = pop.copied.at[jnp.arange(n), write_idx].set(
        jnp.where(is_copy & pop.alive, copy_val, pop.copied[jnp.arange(n), write_idx])
    )
    new_read_after_copy = jnp.where(is_copy, jnp.mod(new_read_after_alloc + 1, max_len), new_read_after_alloc)
    new_write_after_copy = jnp.where(is_copy, jnp.mod(new_write + 1, max_len), new_write)
    new_copied_length = jnp.where(
        is_copy & pop.alive, jnp.minimum(new_copied_length + 1, max_len), new_copied_length
    )

    # JUMP / JUMP_IF_ZERO: set IP = reg0 mod genome_length
    target = jnp.mod(reg0, jnp.maximum(pop.genome_lengths, 1))
    new_ip_jump = jnp.where(op == Op.JUMP, target, pop.ip + 1)
    new_ip_jump = jnp.where(
        (op == Op.JUMP_IF_ZERO) & (reg0 == 0), target, new_ip_jump
    )
    new_ip = jnp.mod(new_ip_jump, jnp.maximum(pop.genome_lengths, 1))

    # EMIT: increases merit
    merit_gain = jnp.where(op == Op.EMIT, 0.05, 0.0)
    # Useful arithmetic also gains small merit
    merit_gain = merit_gain + jnp.where(op == Op.NAND, 0.01, 0.0)

    new_registers = reg.at[:, 0].set(new_reg0).at[:, 1].set(new_reg1).at[:, 2].set(reg2) if cfg.registers >= 3 else reg.at[:, 0].set(new_reg0).at[:, 1].set(new_reg1)

    pop.registers = new_registers
    pop.copied = copied
    pop.copied_length = new_copied_length
    pop.read_head = new_read_after_copy
    pop.write_head = new_write_after_copy
    pop.ip = new_ip
    pop.merit = pop.merit + merit_gain * pop.alive.astype(jnp.float32)
    prev_mask = getattr(pop, "_divide_op_mask", jnp.zeros_like(pop.alive))
    pop._divide_op_mask = prev_mask | ((op == Op.H_DIVIDE) & pop.alive)  # type: ignore[attr-defined]
    return pop


# ---------------------------------------------------------------------------
# Division (replication) at update boundary
# ---------------------------------------------------------------------------


def apply_h_divide(
    pop: AvidaPopulationState,
    cfg: AvidaConfig,
    next_lineage_start: int,
    key: PRNGKey,
) -> AvidaPopulationState:
    """For each organism that just executed H_DIVIDE with enough copied content,
    spawn its offspring into a free slot."""
    n = pop.alive.shape[0]
    div_mask = getattr(pop, "_divide_op_mask", jnp.zeros((n,), dtype=bool))
    # Require at least a handful of copied instructions, but not the full half-genome.
    min_copy = max(1, cfg.genome_length // 8)
    can_divide = div_mask & (pop.copied_length >= min_copy)

    dead = ~pop.alive
    parent_idx = jnp.where(can_divide, jnp.arange(n), n)
    parent_order = jnp.argsort(parent_idx)
    dead_idx = jnp.where(dead, jnp.arange(n), n)
    dead_order = jnp.argsort(dead_idx)
    num_matches = jnp.minimum(jnp.sum(can_divide), jnp.sum(dead))

    match_range = jnp.arange(n)
    use = match_range < num_matches
    parents = jnp.where(use, parent_order, 0)
    children = jnp.where(use, dead_order, 0)

    # Build child genome from parent's copied buffer, with insertion/deletion mutations
    parent_copied = pop.copied[parents]  # [n, max_len]
    parent_len = pop.copied_length[parents]  # [n]

    key, k_ins, k_del = jax.random.split(key, 3)
    insertion = jax.random.bernoulli(k_ins, cfg.insertion_prob, (n,))
    deletion = jax.random.bernoulli(k_del, cfg.deletion_prob, (n,))
    child_len = jnp.clip(
        parent_len + insertion.astype(jnp.int32) - deletion.astype(jnp.int32),
        1,
        cfg.max_genome_length,
    )

    # Write child genome (just adopt parent_copied directly; length controls validity)
    use_b = use[:, None]
    new_genomes = pop.genomes.at[children].set(
        jnp.where(use_b, parent_copied, pop.genomes[children])
    )
    new_lengths = pop.genome_lengths.at[children].set(
        jnp.where(use, child_len, pop.genome_lengths[children])
    )

    # Reset child state
    new_registers = pop.registers.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.registers[children]), pop.registers[children])
    )
    new_memory = pop.memory.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.memory[children]), pop.memory[children])
    )
    new_ip = pop.ip.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.ip[children]))
    new_rh = pop.read_head.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.read_head[children]))
    new_wh = pop.write_head.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.write_head[children]))
    new_copied = pop.copied.at[children].set(
        jnp.where(use_b, jnp.zeros_like(pop.copied[children]), pop.copied[children])
    )
    new_cl = pop.copied_length.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.copied_length[children]))
    new_merit = pop.merit.at[children].set(jnp.where(use, jnp.ones_like(use, dtype=jnp.float32), pop.merit[children]))
    new_age = pop.age.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.age[children]))
    new_alive = pop.alive.at[children].set(jnp.where(use, jnp.ones_like(use), pop.alive[children]))

    # Reset parent's copied buffer so it must alloc again before next divide
    new_copied = new_copied.at[parents].set(
        jnp.where(use_b, jnp.zeros_like(new_copied[parents]), new_copied[parents])
    )
    new_cl = new_cl.at[parents].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), new_cl[parents]))

    new_lineage_ids = next_lineage_start + jnp.arange(n, dtype=jnp.int32)
    new_lineage_id = pop.lineage_id.at[children].set(
        jnp.where(use, new_lineage_ids, pop.lineage_id[children])
    )
    new_parent_id = pop.parent_id.at[children].set(
        jnp.where(use, pop.lineage_id[parents], pop.parent_id[children])
    )
    new_lineage_depth = pop.lineage_depth.at[children].set(
        jnp.where(use, pop.lineage_depth[parents] + 1, pop.lineage_depth[children])
    )

    pop.genomes = new_genomes
    pop.genome_lengths = new_lengths
    pop.registers = new_registers
    pop.memory = new_memory
    pop.ip = new_ip
    pop.read_head = new_rh
    pop.write_head = new_wh
    pop.copied = new_copied
    pop.copied_length = new_cl
    pop.merit = new_merit
    pop.age = new_age
    pop.alive = new_alive
    pop.lineage_id = new_lineage_id
    pop.parent_id = new_parent_id
    pop.lineage_depth = new_lineage_depth
    return pop


def step_avida_population(
    pop: AvidaPopulationState,
    world: WorldState,
    cfg: AvidaConfig,
    next_lineage_start: int,
    key: PRNGKey,
    positions: jax.Array | None = None,
) -> tuple[AvidaPopulationState, int]:
    """Execute cycles_per_update instructions per organism, then resolve divisions.

    Returns (pop, next_lineage_start_after).
    """
    cycles = cfg.cycles_per_update
    pop._divide_op_mask = jnp.zeros_like(pop.alive)  # type: ignore[attr-defined]
    for c in range(cycles):
        key, k_cyc = jax.random.split(key)
        pop = vm_cycle(pop, world, cfg, positions, k_cyc)

    key, k_div = jax.random.split(key)
    pop = apply_h_divide(pop, cfg, next_lineage_start, k_div)
    pop.age = pop.age + pop.alive.astype(jnp.int32)
    next_lineage_start = next_lineage_start + pop.alive.shape[0]
    return pop, next_lineage_start


def mutate_avida_genomes(
    genomes: jax.Array,
    genome_lengths: jax.Array,
    cfg: AvidaConfig,
    key: PRNGKey,
) -> tuple[jax.Array, jax.Array]:
    """Background mutation pass: applies point mutations across all genomes.

    Insertion/deletion is handled at divide time; this function exists for
    occasional radiation-like background mutation.
    """
    k_mask, k_val = jax.random.split(key)
    mask = jax.random.bernoulli(k_mask, cfg.point_mutation_prob, genomes.shape)
    replacement = jax.random.randint(k_val, genomes.shape, 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    return jnp.where(mask, replacement, genomes), genome_lengths
