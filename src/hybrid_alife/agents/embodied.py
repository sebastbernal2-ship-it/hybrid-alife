"""Embodied recurrent neural agents.

Each agent has a flat-parameter GRU-style controller stored per-agent so that
mutation acts directly on the genome tensors. The architecture is intentionally
simple to keep mutation closed-form, but the controller has gated recurrence
(`use_memory=False` ablation forces it into an MLP).

Observation layout:
  patch  : (2r+1)^2 * patch_channels
  sense  : 9 sixth-sense channels (1 + 2 + 1 + 2 + 2 + 3 -- see sensing.sixth_sense_dim)
  prop   : 6 + message_size proprioception channels (pos, energy, age, alive, hidden-norm, message)

Action layout (continuous):
  0..1: dx, dy in [-1, 1]
  2:    eat gate
  3:    attack gate
  4:    reproduce gate
  5:    terraform-resource gate
  6:    terraform-hazard gate
  7:    emit-message gate
  8..:  message vector (message_size)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import (
    EmbodiedConfig,
    EmbodiedPopulationState,
    PRNGKey,
    WorldConfig,
    WorldState,
)
from hybrid_alife.world.env import (
    consume_from_grid,
    gather_patch,
    hazard_damage,
    positions_to_grid,
    scatter_add_to_grid,
)
from hybrid_alife.world.sensing import sample_sixth_sense, sixth_sense_dim


PATCH_CHANNELS_DEFAULT = (
    4  # terrain
    + 3  # resources (worldcfg default)
    + 2  # hazards
    + 2  # flow
    + 1  # curvature
    + 2  # shear
    + 2  # shear_grad
    + 1  # enrichment
    + 2  # lift
    + 2  # concentration
    + 2  # metabolites
)


def patch_channels(world_cfg: WorldConfig) -> int:
    return (
        4
        + world_cfg.resource_channels
        + world_cfg.hazard_channels
        + 2  # flow
        + 1  # curvature
        + 2  # shear
        + 2  # shear_grad
        + 1  # enrichment
        + 2  # lift
        + 2  # concentration
        + world_cfg.metabolite_channels
    )


def embodied_observation_dim(embodied_cfg: EmbodiedConfig, world_cfg: WorldConfig) -> int:
    patch_side = 2 * embodied_cfg.obs_radius + 1
    patch_size = patch_side * patch_side * patch_channels(world_cfg)
    sense = sixth_sense_dim(world_cfg)
    proprio = 6 + embodied_cfg.message_size
    return patch_size + sense + proprio


def embodied_action_dim(cfg: EmbodiedConfig) -> int:
    return 8 + cfg.message_size


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_embodied_population(
    cfg: EmbodiedConfig, world_cfg: WorldConfig, key: PRNGKey
) -> EmbodiedPopulationState:
    """Initialize positions, energies, recurrent state, messages, and genome params."""
    k_pos, k_genome = jax.random.split(key)
    n = cfg.population_size
    positions = jax.random.uniform(k_pos, (n, 2), minval=0.0, maxval=1.0)
    energy = jnp.full((n,), cfg.initial_energy, dtype=jnp.float32)
    age = jnp.zeros((n,), dtype=jnp.int32)
    alive = jnp.ones((n,), dtype=bool)
    hidden = jnp.zeros((n, cfg.hidden_size), dtype=jnp.float32)
    messages = jnp.zeros((n, cfg.message_size), dtype=jnp.float32)
    genomes = initialize_embodied_genomes(cfg, world_cfg, k_genome)
    lineage_id = jnp.arange(n, dtype=jnp.int32)
    parent_id = -jnp.ones((n,), dtype=jnp.int32)
    lineage_depth = jnp.zeros((n,), dtype=jnp.int32)
    action_dim = embodied_action_dim(cfg)
    last_action = jnp.zeros((n, action_dim), dtype=jnp.float32)
    action_history = jnp.zeros((n, cfg.action_history_len), dtype=jnp.int32)
    behavior_descriptor = jnp.zeros((n, 2), dtype=jnp.float32)
    return EmbodiedPopulationState(
        positions=positions,
        energy=energy,
        age=age,
        alive=alive,
        hidden=hidden,
        messages=messages,
        genomes=genomes,
        lineage_id=lineage_id,
        parent_id=parent_id,
        lineage_depth=lineage_depth,
        last_action=last_action,
        action_history=action_history,
        behavior_descriptor=behavior_descriptor,
    )


def initialize_embodied_genomes(
    cfg: EmbodiedConfig, world_cfg: WorldConfig, key: PRNGKey
) -> dict[str, jax.Array]:
    obs_dim = embodied_observation_dim(cfg, world_cfg)
    action_dim = embodied_action_dim(cfg)
    h = cfg.hidden_size
    keys = jax.random.split(key, 7)
    n = cfg.population_size
    return {
        # GRU-like: combined update gate
        "w_xz": 0.1 * jax.random.normal(keys[0], (n, obs_dim, h)),
        "w_hz": 0.1 * jax.random.normal(keys[1], (n, h, h)),
        "b_z": jnp.zeros((n, h)),
        # Candidate
        "w_xh": 0.1 * jax.random.normal(keys[2], (n, obs_dim, h)),
        "w_hh": 0.1 * jax.random.normal(keys[3], (n, h, h)),
        "b_h": jnp.zeros((n, h)),
        # Action head
        "w_act": 0.1 * jax.random.normal(keys[4], (n, h, action_dim)),
        "b_act": 0.01 * jax.random.normal(keys[5], (n, action_dim)),
    }


# ---------------------------------------------------------------------------
# Observation & controller
# ---------------------------------------------------------------------------


def build_world_field(world: WorldState, world_cfg: WorldConfig) -> jax.Array:
    """Concatenate all per-cell channels in a fixed order matching patch_channels()."""
    parts = [
        world.terrain,
        world.resources,
        world.hazards,
        world.flow,
        world.curvature,
        world.shear,
        world.shear_grad,
        world.enrichment,
        world.lift,
        world.concentration,
        world.metabolites,
    ]
    return jnp.concatenate(parts, axis=-1)


def observe_embodied(
    pop: EmbodiedPopulationState,
    world: WorldState,
    embodied_cfg: EmbodiedConfig,
    world_cfg: WorldConfig,
    key: PRNGKey,
) -> jax.Array:
    """Build observation tensor [N, obs_dim]."""
    h, w = world_cfg.height, world_cfg.width
    field = build_world_field(world, world_cfg)
    if embodied_cfg.blind:
        n = pop.positions.shape[0]
        side = 2 * embodied_cfg.obs_radius + 1
        patch_flat = jnp.zeros((n, side * side * patch_channels(world_cfg)), dtype=jnp.float32)
        sense = jnp.zeros((n, sixth_sense_dim(world_cfg)), dtype=jnp.float32)
    else:
        patch_flat = gather_patch(
            field, pop.positions, embodied_cfg.obs_radius, h, w, world_cfg.toroidal
        )
        sense = sample_sixth_sense(
            world,
            pop.positions,
            pop.alive,
            pop.messages,
            world_cfg,
            embodied_cfg,
            key,
        )

    proprio = jnp.concatenate(
        [
            pop.positions,
            pop.energy[:, None],
            pop.age[:, None].astype(jnp.float32),
            pop.alive[:, None].astype(jnp.float32),
            jnp.linalg.norm(pop.hidden, axis=-1, keepdims=True),
            pop.messages if embodied_cfg.use_comms else jnp.zeros_like(pop.messages),
        ],
        axis=-1,
    )

    return jnp.concatenate([patch_flat, sense, proprio], axis=-1).astype(jnp.float32)


def act_embodied(
    pop: EmbodiedPopulationState, obs: jax.Array, cfg: EmbodiedConfig
) -> tuple[jax.Array, EmbodiedPopulationState]:
    """GRU-like step. If cfg.use_memory is False, hidden is reset every step (MLP ablation)."""
    g = pop.genomes
    z = jax.nn.sigmoid(
        jnp.einsum("no,noh->nh", obs, g["w_xz"])
        + jnp.einsum("nh,nhk->nk", pop.hidden, g["w_hz"])
        + g["b_z"]
    )
    cand = jnp.tanh(
        jnp.einsum("no,noh->nh", obs, g["w_xh"])
        + jnp.einsum("nh,nhk->nk", pop.hidden, g["w_hh"])
        + g["b_h"]
    )
    new_hidden = (1.0 - z) * pop.hidden + z * cand
    if not cfg.use_memory:
        new_hidden = cand  # no recurrent carry
    raw = jnp.einsum("nh,nha->na", new_hidden, g["w_act"]) + g["b_act"]
    # Movement to [-1, 1]; gates to [0, 1]; message linear
    move = jnp.tanh(raw[..., 0:2])
    gates = jax.nn.sigmoid(raw[..., 2:8])
    msg = raw[..., 8:]
    actions = jnp.concatenate([move, gates, msg], axis=-1)
    pop.hidden = new_hidden
    return actions, pop


# ---------------------------------------------------------------------------
# Mutation & reproduction
# ---------------------------------------------------------------------------


def mutate_embodied_genomes(
    genomes: dict[str, jax.Array], cfg: EmbodiedConfig, key: PRNGKey
) -> dict[str, jax.Array]:
    """Elementwise Gaussian perturbation under Bernoulli mask, per-individual."""
    keys = jax.random.split(key, len(genomes))
    out: dict[str, jax.Array] = {}
    for subkey, (name, value) in zip(keys, genomes.items(), strict=True):
        k_mask, k_noise = jax.random.split(subkey)
        mask = jax.random.bernoulli(k_mask, cfg.genome_mutation_prob, value.shape)
        noise = cfg.genome_mutation_std * jax.random.normal(k_noise, value.shape)
        out[name] = value + mask * noise
    return out


# ---------------------------------------------------------------------------
# World-coupled action application
# ---------------------------------------------------------------------------


def apply_embodied_actions(
    pop: EmbodiedPopulationState,
    actions: jax.Array,
    world: WorldState,
    embodied_cfg: EmbodiedConfig,
    world_cfg: WorldConfig,
    key: PRNGKey,
) -> tuple[EmbodiedPopulationState, WorldState, jax.Array]:
    """Apply actions to update positions, energy, lifecycle, world resources, messages.

    Returns (new pop, new world, per-agent action-index for entropy tracking).
    """
    h, w = world_cfg.height, world_cfg.width
    alive_f = pop.alive.astype(jnp.float32)

    move = actions[..., 0:2] * 0.05  # max 5% of world per step
    eat = actions[..., 2]
    attack = actions[..., 3]
    repro = actions[..., 4]
    terra_r = actions[..., 5]
    terra_h = actions[..., 6]
    emit = actions[..., 7]
    msg_vec = actions[..., 8:]

    # Move with toroidal wrap
    new_pos = pop.positions + move * alive_f[:, None]
    if world_cfg.toroidal:
        new_pos = jnp.mod(new_pos, 1.0)
    else:
        new_pos = jnp.clip(new_pos, 0.0, 0.999)

    # Eat: drain resources at cell
    eat_rate = 0.2 * eat * alive_f
    grid = positions_to_grid(new_pos, h, w)
    cell_resources = world.resources[grid[:, 0], grid[:, 1]]  # [N, C]
    take_per_channel = cell_resources * eat_rate[:, None]
    new_resources = world.resources.at[grid[:, 0], grid[:, 1]].add(-take_per_channel)
    new_resources = jnp.clip(new_resources, 0.0, 1.0)
    energy_gained = jnp.sum(take_per_channel, axis=-1) * 5.0  # scaling

    # Hazard damage (always applies, gated by attack to "fight"=halve damage)
    haz = hazard_damage(world.hazards, new_pos, pop.alive, h, w)
    damage = haz * (1.0 - 0.5 * attack)

    # Terraform: deposit positive concentration if terra_r, negative if terra_h
    deposit = jnp.stack(
        [(terra_r - terra_h) * 0.5 * alive_f, jnp.zeros_like(terra_r)], axis=-1
    )
    new_conc = scatter_add_to_grid(world.concentration, new_pos, deposit, pop.alive, h, w)

    # Metabolite deposit proportional to energy expenditure
    metab_scalar = (0.01 + 0.05 * eat) * alive_f  # [N]
    metab = jnp.broadcast_to(metab_scalar[:, None], (metab_scalar.shape[0], world_cfg.metabolite_channels))
    new_metab = scatter_add_to_grid(world.metabolites, new_pos, metab, pop.alive, h, w)

    # Hazard accumulation from attacks (attack deposits hazard locally)
    hazard_scalar = 0.02 * attack * alive_f  # [N]
    hazard_dep = jnp.broadcast_to(hazard_scalar[:, None], (hazard_scalar.shape[0], world_cfg.hazard_channels))
    new_hazards = scatter_add_to_grid(world.hazards, new_pos, hazard_dep, pop.alive, h, w)
    new_hazards = jnp.clip(new_hazards, 0.0, 1.0)

    # Emit messages (gated)
    new_msg = pop.messages * 0.5 + emit[:, None] * msg_vec * alive_f[:, None]
    if not embodied_cfg.use_comms:
        new_msg = jnp.zeros_like(new_msg)

    # Energy update
    move_cost = jnp.linalg.norm(move, axis=-1) * 0.5
    new_energy = (
        pop.energy
        + energy_gained
        - damage
        - move_cost
        - embodied_cfg.basal_metabolic_cost
        - 0.05 * repro
    )

    # Reproduction handled in apply_reproduction; here we only zero-out parents' energy cost when triggered.

    # Mark dead (energy<=0)
    new_alive = pop.alive & (new_energy > 0.0)
    new_energy = jnp.where(new_alive, new_energy, 0.0)
    new_age = pop.age + new_alive.astype(jnp.int32)

    # Behavior descriptor: running mean of (mean speed, eat rate)
    speed = jnp.linalg.norm(move, axis=-1)
    bd_speed = 0.95 * pop.behavior_descriptor[:, 0] + 0.05 * speed
    bd_eat = 0.95 * pop.behavior_descriptor[:, 1] + 0.05 * eat
    new_bd = jnp.stack([bd_speed, bd_eat], axis=-1)

    # Action index: argmax over gate-like actions (move-x-pos, move-x-neg, move-y-pos, move-y-neg,
    # eat, attack, repro, terra_r, terra_h, emit)
    scores = jnp.stack(
        [
            jnp.maximum(move[:, 0], 0.0),
            jnp.maximum(-move[:, 0], 0.0),
            jnp.maximum(move[:, 1], 0.0),
            jnp.maximum(-move[:, 1], 0.0),
            eat,
            attack,
            repro,
            terra_r,
            terra_h,
            emit,
        ],
        axis=-1,
    )
    action_idx = jnp.argmax(scores, axis=-1).astype(jnp.int32)
    new_history = jnp.concatenate(
        [pop.action_history[:, 1:], action_idx[:, None]], axis=-1
    )

    _ = key
    new_world = WorldState(
        terrain=world.terrain,
        resources=new_resources,
        hazards=new_hazards,
        flow=world.flow,
        curvature=world.curvature,
        shear=world.shear,
        shear_grad=world.shear_grad,
        enrichment=world.enrichment,
        lift=world.lift,
        concentration=new_conc,
        metabolites=new_metab,
        occupancy=world.occupancy,
        time=world.time,
    )

    pop.positions = new_pos
    pop.energy = new_energy
    pop.alive = new_alive
    pop.age = new_age
    pop.messages = new_msg
    pop.last_action = actions
    pop.action_history = new_history
    pop.behavior_descriptor = new_bd
    return pop, new_world, repro


def apply_reproduction(
    pop: EmbodiedPopulationState,
    repro_gate: jax.Array,
    cfg: EmbodiedConfig,
    next_lineage_start: int,
    key: PRNGKey,
) -> EmbodiedPopulationState:
    """Asexual reproduction into the first available dead slots.

    A parent reproduces iff alive, energy >= threshold, and repro_gate > 0.5.
    The parent loses reproduction_energy_cost. The child inherits mutated genomes.
    """
    n = pop.alive.shape[0]
    can_reproduce = (
        pop.alive & (pop.energy >= cfg.reproduce_energy_threshold) & (repro_gate > 0.5)
    )

    dead = ~pop.alive  # [N]
    # Match each reproducing parent to a dead slot greedily.
    parent_idx = jnp.where(can_reproduce, jnp.arange(n), n)  # n = sentinel
    parent_order = jnp.argsort(parent_idx)  # parents first, sentinels last
    dead_idx = jnp.where(dead, jnp.arange(n), n)
    dead_order = jnp.argsort(dead_idx)
    num_matches = jnp.minimum(jnp.sum(can_reproduce), jnp.sum(dead))

    match_range = jnp.arange(n)
    use = match_range < num_matches
    parents = jnp.where(use, parent_order, 0)
    children = jnp.where(use, dead_order, 0)

    # Mutate parent's genomes per-slot and write into child slots.
    keys = jax.random.split(key, len(pop.genomes))
    new_genomes: dict[str, jax.Array] = {}
    for subkey, (name, value) in zip(keys, pop.genomes.items(), strict=True):
        k_mask, k_noise = jax.random.split(subkey)
        parent_g = value[parents]
        mask = jax.random.bernoulli(k_mask, cfg.genome_mutation_prob, parent_g.shape)
        noise = cfg.genome_mutation_std * jax.random.normal(k_noise, parent_g.shape)
        child_g = parent_g + mask * noise
        # write only into actual children (use_mask)
        use_b = use[:, None]
        if value.ndim == 3:
            use_b = use[:, None, None]
        elif value.ndim == 2:
            use_b = use[:, None]
        # we need to update value[children[i]] when use[i]
        updated = value.at[children].set(jnp.where(use_b, child_g, value[children]))
        new_genomes[name] = updated

    # Update child state arrays
    child_positions = pop.positions[parents]
    new_positions = pop.positions.at[children].set(
        jnp.where(use[:, None], child_positions, pop.positions[children])
    )

    parent_energy_after = pop.energy[parents] - cfg.reproduction_energy_cost
    child_energy = cfg.initial_energy * 0.5
    new_energy = pop.energy
    new_energy = new_energy.at[parents].set(
        jnp.where(use, parent_energy_after, new_energy[parents])
    )
    new_energy = new_energy.at[children].set(
        jnp.where(use, child_energy, new_energy[children])
    )

    new_alive = pop.alive.at[children].set(
        jnp.where(use, jnp.ones_like(use), pop.alive[children])
    )
    new_age = pop.age.at[children].set(jnp.where(use, jnp.zeros_like(use, dtype=jnp.int32), pop.age[children]))
    new_hidden = pop.hidden.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.hidden[children]), pop.hidden[children])
    )
    new_messages = pop.messages.at[children].set(
        jnp.where(use[:, None], jnp.zeros_like(pop.messages[children]), pop.messages[children])
    )

    # Lineage
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
    pop.positions = new_positions
    pop.energy = new_energy
    pop.alive = new_alive
    pop.age = new_age
    pop.hidden = new_hidden
    pop.messages = new_messages
    pop.lineage_id = new_lineage_id
    pop.parent_id = new_parent_id
    pop.lineage_depth = new_lineage_depth
    return pop
