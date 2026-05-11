"""Avida-inspired instruction-genome virtual machine.

Vectorized over the population. Each digital organism has:
  - a genome (fixed-capacity int32 array of length max_genome_length)
  - an actual genome_length (logical length used)
  - registers, scratch memory, instruction pointer
  - read/write heads pointing into the parent and an offspring buffer
  - merit (proportional fitness from useful work)

Replication is the classic Avida pattern:

  H_ALLOC  : open offspring buffer of size max_genome_length
  H_COPY   : copy ``genome[read_head]`` -> ``copied[write_head]`` with point
             mutation, advancing both heads.
  H_DIVIDE : if a chunk of contiguous content has been copied, spawn into a
             free slot, optionally with insertion / deletion / duplication
             mutations applied to the child genome.

Logic-task rewards
~~~~~~~~~~~~~~~~~~
We follow the classic Avida task pool: each time an organism ``EMIT``s a
register value that equals a target boolean combination of its two most
recent ``LOAD_ENV`` inputs ``a, b``, the corresponding task is credited and
its reward is added to merit. Tasks are deduplicated per organism per
generation via a bitfield so an organism cannot loop the cheapest task
forever.

CPU-cycle allocation
~~~~~~~~~~~~~~~~~~~~
At the start of each update we compute a per-organism cycle count of::

    base = cfg.cycles_per_update
    extra = base * (merit / merit_floor) ** merit_cycle_exponent

clamped to ``cfg.max_cycles_per_update``. Each inner cycle skips organisms
that have used up their budget — implemented as an alive mask reset for
that cycle only. This rewards productive organisms with extra compute,
matching the original Avida design.
"""

from __future__ import annotations

from enum import IntEnum

import jax
import jax.numpy as jnp

from hybrid_alife.types import AvidaConfig, AvidaPopulationState, PRNGKey, WorldState
from hybrid_alife.world.env import positions_to_grid, scatter_add_to_grid


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
# Logic-task definitions
# ---------------------------------------------------------------------------


# Order matters: this defines the bit index in tasks_completed.
TASK_NAMES = (
    "not",
    "nand",
    "and",
    "orn",
    "or",
    "andn",
    "nor",
    "xor",
    "equ",
)


def _logic_task_outputs(a: jax.Array, b: jax.Array) -> jax.Array:
    """Return ``[N, 9]`` int32 array of expected outputs for each task."""
    not_a = jnp.bitwise_not(a)
    nand = jnp.bitwise_not(jnp.bitwise_and(a, b))
    and_ = jnp.bitwise_and(a, b)
    orn = jnp.bitwise_or(a, jnp.bitwise_not(b))
    or_ = jnp.bitwise_or(a, b)
    andn = jnp.bitwise_and(a, jnp.bitwise_not(b))
    nor = jnp.bitwise_not(jnp.bitwise_or(a, b))
    xor = jnp.bitwise_xor(a, b)
    equ = jnp.bitwise_not(jnp.bitwise_xor(a, b))
    return jnp.stack([not_a, nand, and_, orn, or_, andn, nor, xor, equ], axis=-1)


def _task_reward_vector(cfg: AvidaConfig) -> jax.Array:
    return jnp.asarray(
        [
            cfg.task_reward_not,
            cfg.task_reward_nand,
            cfg.task_reward_and,
            cfg.task_reward_orn,
            cfg.task_reward_or,
            cfg.task_reward_andn,
            cfg.task_reward_nor,
            cfg.task_reward_xor,
            cfg.task_reward_equ,
        ],
        dtype=jnp.float32,
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_avida_population(cfg: AvidaConfig, key: PRNGKey) -> AvidaPopulationState:
    """Initialize digital organisms with a seeded self-replicator template."""
    n = cfg.population_size
    max_len = cfg.max_genome_length
    k_rand, k_seed = jax.random.split(key)
    genomes = jax.random.randint(k_rand, (n, max_len), 0, INSTRUCTION_COUNT, dtype=jnp.int32)

    template = jnp.array(
        [Op.H_ALLOC, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_COPY, Op.H_DIVIDE],
        dtype=jnp.int32,
    )
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
    tasks_completed = jnp.zeros((n,), dtype=jnp.int32)
    last_input_a = jnp.zeros((n,), dtype=jnp.int32)
    last_input_b = jnp.zeros((n,), dtype=jnp.int32)
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
        tasks_completed=tasks_completed,
        last_input_a=last_input_a,
        last_input_b=last_input_b,
    )


def _ensure_task_fields(pop: AvidaPopulationState) -> AvidaPopulationState:
    """Backfill new fields for AvidaPopulationState loaded from older checkpoints."""
    n = pop.alive.shape[0]
    if pop.tasks_completed is None:
        pop.tasks_completed = jnp.zeros((n,), dtype=jnp.int32)
    if pop.last_input_a is None:
        pop.last_input_a = jnp.zeros((n,), dtype=jnp.int32)
    if pop.last_input_b is None:
        pop.last_input_b = jnp.zeros((n,), dtype=jnp.int32)
    return pop


# ---------------------------------------------------------------------------
# Per-cycle execution
# ---------------------------------------------------------------------------


def vm_cycle(
    pop: AvidaPopulationState,
    world: WorldState,
    cfg: AvidaConfig,
    positions: jax.Array | None,
    key: PRNGKey,
    cycle_mask: jax.Array | None = None,
) -> AvidaPopulationState:
    """Execute one vectorized instruction per organism.

    ``cycle_mask`` (optional [N] bool) gates organisms that have exhausted
    their per-update cycle budget. Masked organisms skip the cycle entirely
    (no IP advance, no merit gain, no state change) which lets us implement
    merit-based CPU-cycle allocation without varying loop counts per agent.
    """
    pop = _ensure_task_fields(pop)
    n = pop.genomes.shape[0]
    max_len = cfg.max_genome_length

    if cycle_mask is None:
        cycle_mask = jnp.ones((n,), dtype=bool)
    active = pop.alive & cycle_mask

    idx = jnp.mod(pop.ip, jnp.maximum(pop.genome_lengths, 1))
    op = jnp.take_along_axis(pop.genomes, idx[:, None], axis=1)[:, 0]

    reg = pop.registers
    reg0 = reg[:, 0]
    reg1 = reg[:, 1]
    reg2 = reg[:, 2] if cfg.registers >= 3 else jnp.zeros_like(reg0)

    # ---- ALU --------------------------------------------------------------
    new_reg0 = reg0
    new_reg0 = jnp.where(op == Op.INC, new_reg0 + 1, new_reg0)
    new_reg0 = jnp.where(op == Op.DEC, new_reg0 - 1, new_reg0)
    new_reg0 = jnp.where(op == Op.ADD, reg0 + reg1, new_reg0)
    new_reg0 = jnp.where(op == Op.NAND, jnp.bitwise_not(jnp.bitwise_and(reg0, reg1)), new_reg0)
    new_reg0 = jnp.where(op == Op.SHIFT_L, jnp.left_shift(reg0, 1), new_reg0)
    new_reg0 = jnp.where(op == Op.SHIFT_R, jnp.right_shift(reg0, 1), new_reg0)

    # ---- Environment IO --------------------------------------------------
    # We sample two distinct env values per organism (a from enrichment,
    # b from concentration). Both are quantized to int32 for the bitwise
    # logic tasks.
    if positions is not None and positions.shape[0] == n:
        h, w = world.enrichment.shape[0], world.enrichment.shape[1]
        grid = positions_to_grid(positions, h, w)
        env_a = (world.enrichment[grid[:, 0], grid[:, 1], 0] * 1000.0).astype(jnp.int32)
        env_b = (world.concentration[grid[:, 0], grid[:, 1], 0] * 1000.0).astype(jnp.int32)
    else:
        env_a = jnp.full((n,), int(jnp.mean(world.enrichment) * 1000.0), dtype=jnp.int32)
        env_b = jnp.full((n,), int(jnp.mean(world.concentration) * 1000.0), dtype=jnp.int32)

    is_loadenv = op == Op.LOAD_ENV
    new_reg0 = jnp.where(is_loadenv, env_a, new_reg0)
    new_reg1 = jnp.where(is_loadenv, env_b, reg1)
    new_input_a = jnp.where(is_loadenv, env_a, pop.last_input_a)
    new_input_b = jnp.where(is_loadenv, env_b, pop.last_input_b)

    # ---- Head manipulation ----------------------------------------------
    new_read = jnp.where(op == Op.SET_READ, jnp.mod(reg0, max_len), pop.read_head)
    new_write = jnp.where(op == Op.SET_WRITE, jnp.mod(reg0, max_len), pop.write_head)

    is_alloc = op == Op.H_ALLOC
    new_copied_length = jnp.where(is_alloc, 0, pop.copied_length)
    new_write = jnp.where(is_alloc, 0, new_write)
    new_read = jnp.where(is_alloc, 0, new_read)

    is_copy = op == Op.H_COPY
    src_idx = jnp.mod(new_read, jnp.maximum(pop.genome_lengths, 1))
    src_val = jnp.take_along_axis(pop.genomes, src_idx[:, None], axis=1)[:, 0]
    key, k_mut, k_val = jax.random.split(key, 3)
    mut_mask = jax.random.bernoulli(k_mut, cfg.point_mutation_prob, (n,))
    mut_val = jax.random.randint(k_val, (n,), 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    copy_val = jnp.where(mut_mask, mut_val, src_val)
    write_idx = jnp.mod(new_write, max_len)
    copied = pop.copied.at[jnp.arange(n), write_idx].set(
        jnp.where(is_copy & active, copy_val, pop.copied[jnp.arange(n), write_idx])
    )
    new_read = jnp.where(is_copy, jnp.mod(new_read + 1, max_len), new_read)
    new_write = jnp.where(is_copy, jnp.mod(new_write + 1, max_len), new_write)
    new_copied_length = jnp.where(
        is_copy & active, jnp.minimum(new_copied_length + 1, max_len), new_copied_length
    )

    # ---- Control flow ----------------------------------------------------
    target = jnp.mod(reg0, jnp.maximum(pop.genome_lengths, 1))
    new_ip_jump = jnp.where(op == Op.JUMP, target, pop.ip + 1)
    new_ip_jump = jnp.where(
        (op == Op.JUMP_IF_ZERO) & (reg0 == 0), target, new_ip_jump
    )
    new_ip = jnp.mod(new_ip_jump, jnp.maximum(pop.genome_lengths, 1))

    # ---- EMIT + logic-task scoring --------------------------------------
    # EMIT publishes register 0. If the emitted value matches a logic-task
    # output for (last_input_a, last_input_b), the bit is credited and the
    # task reward is added to merit. We dedupe per-organism.
    expected = _logic_task_outputs(new_input_a, new_input_b)  # [N, 9]
    is_emit = op == Op.EMIT
    match = (expected == new_reg0[:, None])  # [N, 9]
    bits = jnp.arange(9, dtype=jnp.int32)
    already = (jnp.bitwise_and(pop.tasks_completed[:, None], jnp.left_shift(1, bits)) != 0)
    newly = is_emit[:, None] & match & (~already) & active[:, None]
    rewards = _task_reward_vector(cfg)
    merit_gain = jnp.sum(newly.astype(jnp.float32) * rewards[None, :], axis=-1)
    new_tasks = pop.tasks_completed | jnp.sum(
        newly.astype(jnp.int32) * jnp.left_shift(1, bits)[None, :], axis=-1
    )

    # Small per-instruction merit hints to maintain backwards-compatible
    # NAND/EMIT incentives even when no IO is hooked up.
    merit_gain = merit_gain + jnp.where(is_emit, 0.05, 0.0)
    merit_gain = merit_gain + jnp.where(op == Op.NAND, 0.01, 0.0)

    # ---- Commit ----------------------------------------------------------
    if cfg.registers >= 3:
        new_registers = reg.at[:, 0].set(new_reg0).at[:, 1].set(new_reg1).at[:, 2].set(reg2)
    else:
        new_registers = reg.at[:, 0].set(new_reg0).at[:, 1].set(new_reg1)

    # Active gate: organisms that aren't on this cycle should keep all state.
    active_f = active.astype(jnp.float32)
    active_i = active.astype(jnp.int32)

    pop.registers = jnp.where(active[:, None], new_registers, pop.registers)
    pop.copied = jnp.where(active[:, None], copied, pop.copied)
    pop.copied_length = jnp.where(active, new_copied_length, pop.copied_length)
    pop.read_head = jnp.where(active, new_read, pop.read_head)
    pop.write_head = jnp.where(active, new_write, pop.write_head)
    pop.ip = jnp.where(active, new_ip, pop.ip)
    pop.merit = pop.merit + merit_gain * active_f
    pop.tasks_completed = jnp.where(active, new_tasks, pop.tasks_completed)
    pop.last_input_a = jnp.where(active, new_input_a, pop.last_input_a)
    pop.last_input_b = jnp.where(active, new_input_b, pop.last_input_b)
    _ = active_i  # silence unused-warning style
    prev_mask = getattr(pop, "_divide_op_mask", jnp.zeros_like(pop.alive))
    pop._divide_op_mask = prev_mask | ((op == Op.H_DIVIDE) & active)  # type: ignore[attr-defined]
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
    """For each organism that just executed H_DIVIDE with enough copied
    content, spawn its offspring into a free slot."""
    pop = _ensure_task_fields(pop)
    n = pop.alive.shape[0]
    div_mask = getattr(pop, "_divide_op_mask", jnp.zeros((n,), dtype=bool))
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

    # ---- Child genome construction --------------------------------------
    parent_copied = pop.copied[parents]
    parent_len = pop.copied_length[parents]

    k_ins, k_del, k_dup, k_pos = jax.random.split(key, 4)
    insertion = jax.random.bernoulli(k_ins, cfg.insertion_prob, (n,))
    deletion = jax.random.bernoulli(k_del, cfg.deletion_prob, (n,))
    duplication = jax.random.bernoulli(k_dup, cfg.duplication_prob, (n,))

    # Apply duplication: pick a random window [start, end] in parent_copied
    # and repeat it, bounded by max_genome_length.
    max_len = cfg.max_genome_length
    dup_len = jnp.clip(parent_len // 4, 1, max_len)
    dup_start_raw = jax.random.randint(k_pos, (n,), 0, max_len, dtype=jnp.int32)
    dup_start = jnp.mod(dup_start_raw, jnp.maximum(parent_len, 1))

    arange = jnp.arange(max_len)
    # Build duplication-augmented genome lazily: we only duplicate when the
    # flag is set; otherwise we keep parent_copied unchanged.
    dup_offset = (arange[None, :] - parent_len[:, None])  # >= 0 after parent end
    in_dup_window = (dup_offset >= 0) & (dup_offset < dup_len[:, None])
    dup_src_idx = jnp.mod(dup_start[:, None] + jnp.clip(dup_offset, 0, dup_len[:, None] - 1), jnp.maximum(parent_len[:, None], 1))
    dup_vals = jnp.take_along_axis(parent_copied, dup_src_idx, axis=1)
    augmented = jnp.where(duplication[:, None] & in_dup_window, dup_vals, parent_copied)

    child_len = parent_len.astype(jnp.int32)
    child_len = child_len + insertion.astype(jnp.int32) - deletion.astype(jnp.int32)
    child_len = child_len + (duplication.astype(jnp.int32) * dup_len)
    child_len = jnp.clip(child_len, cfg.min_genome_length, cfg.max_genome_length)

    use_b = use[:, None]
    new_genomes = pop.genomes.at[children].set(
        jnp.where(use_b, augmented, pop.genomes[children])
    )
    new_lengths = pop.genome_lengths.at[children].set(
        jnp.where(use, child_len, pop.genome_lengths[children])
    )

    # ---- Reset child state ----------------------------------------------
    zero_int = jnp.zeros_like(use, dtype=jnp.int32)
    one_bool = jnp.ones_like(use)

    new_registers = pop.registers.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.registers[children]), pop.registers[children])
    )
    new_memory = pop.memory.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.memory[children]), pop.memory[children])
    )
    new_ip = pop.ip.at[children].set(jnp.where(use, zero_int, pop.ip[children]))
    new_rh = pop.read_head.at[children].set(jnp.where(use, zero_int, pop.read_head[children]))
    new_wh = pop.write_head.at[children].set(jnp.where(use, zero_int, pop.write_head[children]))
    new_copied = pop.copied.at[children].set(
        jnp.where(use_b, jnp.zeros_like(pop.copied[children]), pop.copied[children])
    )
    new_cl = pop.copied_length.at[children].set(jnp.where(use, zero_int, pop.copied_length[children]))
    new_merit = pop.merit.at[children].set(
        jnp.where(use, jnp.ones_like(use, dtype=jnp.float32), pop.merit[children])
    )
    new_age = pop.age.at[children].set(jnp.where(use, zero_int, pop.age[children]))
    new_alive = pop.alive.at[children].set(jnp.where(use, one_bool, pop.alive[children]))
    new_tasks = pop.tasks_completed.at[children].set(jnp.where(use, zero_int, pop.tasks_completed[children]))

    # Reset parent buffer so it must alloc again
    new_copied = new_copied.at[parents].set(
        jnp.where(use_b, jnp.zeros_like(new_copied[parents]), new_copied[parents])
    )
    new_cl = new_cl.at[parents].set(jnp.where(use, zero_int, new_cl[parents]))

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
    pop.tasks_completed = new_tasks
    pop.lineage_id = new_lineage_id
    pop.parent_id = new_parent_id
    pop.lineage_depth = new_lineage_depth
    return pop


# ---------------------------------------------------------------------------
# Update loop: merit-based CPU allocation + division
# ---------------------------------------------------------------------------


def _cycle_budget(merit: jax.Array, cfg: AvidaConfig) -> jax.Array:
    """Return per-organism integer cycle budget for this update."""
    base = jnp.float32(cfg.cycles_per_update)
    if cfg.merit_cycle_exponent == 0.0:
        budget = jnp.full_like(merit, base)
    else:
        ratio = jnp.maximum(merit, 0.0) / jnp.maximum(cfg.merit_floor, 1e-6)
        budget = base + base * jnp.power(ratio, cfg.merit_cycle_exponent)
    budget = jnp.clip(budget, 1.0, float(cfg.max_cycles_per_update))
    return budget.astype(jnp.int32)


def step_avida_population(
    pop: AvidaPopulationState,
    world: WorldState,
    cfg: AvidaConfig,
    next_lineage_start: int,
    key: PRNGKey,
    positions: jax.Array | None = None,
) -> tuple[AvidaPopulationState, int]:
    """Execute the update: a number of vm cycles bounded by merit, then divide.

    The outer loop count is fixed at ``max_cycles_per_update`` so that the
    function shape remains static-friendly under jit, but per-organism
    ``cycle_mask`` gates retire organisms once they hit their budget.
    """
    pop = _ensure_task_fields(pop)
    pop._divide_op_mask = jnp.zeros_like(pop.alive)  # type: ignore[attr-defined]

    budget = _cycle_budget(pop.merit, cfg)
    max_cycles = int(cfg.max_cycles_per_update)
    for c in range(max_cycles):
        key, k_cyc = jax.random.split(key)
        cycle_mask = budget > c
        pop = vm_cycle(pop, world, cfg, positions, k_cyc, cycle_mask)

    key, k_div = jax.random.split(key)
    pop = apply_h_divide(pop, cfg, next_lineage_start, k_div)
    pop.age = pop.age + pop.alive.astype(jnp.int32)

    # Cross-branch coupling: digital organisms consume metabolites at their
    # mapped position and deposit a fraction back as concentration. This
    # creates an actual resource-loop between the two branches.
    if positions is not None and positions.shape[0] == pop.alive.shape[0]:
        h, w = world.metabolites.shape[0], world.metabolites.shape[1]
        if cfg.metabolite_uptake > 0.0:
            grid = positions_to_grid(positions, h, w)
            available = world.metabolites[grid[:, 0], grid[:, 1]]
            take = available * cfg.metabolite_uptake * pop.alive.astype(jnp.float32)[:, None]
            world.metabolites = world.metabolites.at[grid[:, 0], grid[:, 1]].add(-take)
            world.metabolites = jnp.clip(world.metabolites, 0.0, 1.0)
            # Add a small merit gain for harvesting metabolites.
            pop.merit = pop.merit + jnp.sum(take, axis=-1) * 5.0
        if cfg.metabolite_deposit > 0.0:
            deposit = jnp.broadcast_to(
                (cfg.metabolite_deposit * pop.alive.astype(jnp.float32))[:, None],
                (pop.alive.shape[0], world.concentration.shape[-1]),
            )
            world.concentration = scatter_add_to_grid(
                world.concentration, positions, deposit, pop.alive, h, w
            )

    next_lineage_start = next_lineage_start + pop.alive.shape[0]
    return pop, next_lineage_start


def mutate_avida_genomes(
    genomes: jax.Array,
    genome_lengths: jax.Array,
    cfg: AvidaConfig,
    key: PRNGKey,
) -> tuple[jax.Array, jax.Array]:
    """Background mutation pass: applies point mutations across all genomes."""
    k_mask, k_val = jax.random.split(key)
    mask = jax.random.bernoulli(k_mask, cfg.point_mutation_prob, genomes.shape)
    replacement = jax.random.randint(k_val, genomes.shape, 0, INSTRUCTION_COUNT, dtype=jnp.int32)
    return jnp.where(mask, replacement, genomes), genome_lengths
