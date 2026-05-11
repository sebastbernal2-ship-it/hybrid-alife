"""Core dataclasses and typed state containers for hybrid-alife."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import jax

Array = jax.Array
PRNGKey = Array


class Branch(IntEnum):
    EMBODIED = 0
    AVIDA = 1


class EmbodiedActionIndex(IntEnum):
    MOVE_X = 0
    MOVE_Y = 1
    EAT = 2
    ATTACK = 3
    REPRODUCE = 4
    TERRAFORM_RESOURCE = 5
    TERRAFORM_HAZARD = 6
    EMIT_MESSAGE = 7


@dataclass(frozen=True)
class WorldConfig:
    width: int
    height: int
    toroidal: bool
    resource_channels: int
    hazard_channels: int
    max_agents: int
    max_digital_organisms: int
    flow_noise_std: float
    concentration_decay: float
    concentration_diffusion: float
    drifting: bool = False
    drift_speed: float = 0.02
    sensory_noise_std: float = 0.05
    resource_regen: float = 0.005
    hazard_decay: float = 0.99
    metabolite_channels: int = 2
    metabolite_decay: float = 0.95
    metabolite_diffusion: float = 0.05


@dataclass(frozen=True)
class EmbodiedConfig:
    enabled: bool
    population_size: int
    obs_radius: int
    hidden_size: int
    message_size: int
    genome_mutation_std: float
    genome_mutation_prob: float
    initial_energy: float
    reproduce_energy_threshold: float
    reproduction_energy_cost: float
    basal_metabolic_cost: float
    use_memory: bool = True
    use_comms: bool = True
    blind: bool = False
    shuffle_sixth_sense: bool = False
    true_sixth_sense: bool = True
    action_history_len: int = 16
    # Controller-level knobs (new): keep backwards-compatible defaults.
    stochastic_actions: bool = False
    action_temperature: float = 1.0
    message_gate_threshold: float = 0.5
    # Action cost model: every gated action consumes a fixed amount of energy
    # in addition to the basal cost. Set scales to 0.0 for the old behavior.
    cost_move_scale: float = 0.5
    cost_eat: float = 0.01
    cost_attack: float = 0.08
    cost_reproduce: float = 0.05
    cost_terraform: float = 0.03
    cost_emit: float = 0.02
    # Branch coupling: how strongly metabolites deposit / poison the world.
    metabolite_deposit_scale: float = 1.0
    hazard_deposit_scale: float = 1.0


@dataclass(frozen=True)
class AvidaConfig:
    enabled: bool
    population_size: int
    genome_length: int
    max_genome_length: int
    registers: int
    memory_size: int
    cycles_per_update: int
    point_mutation_prob: float
    insertion_prob: float
    deletion_prob: float
    # Extended VM knobs. All defaults preserve existing behaviour numerically.
    min_genome_length: int = 4
    duplication_prob: float = 0.0
    # Merit-based CPU-cycle allocation: each organism gets cycles_per_update *
    # (merit / merit_floor)^merit_cycle_exponent extra cycles, clamped.
    merit_cycle_exponent: float = 0.5
    merit_floor: float = 1.0
    max_cycles_per_update: int = 32
    # Per-task merit rewards (single-shot, deduplicated).
    task_reward_not: float = 1.0
    task_reward_nand: float = 1.0
    task_reward_and: float = 2.0
    task_reward_orn: float = 2.0
    task_reward_or: float = 3.0
    task_reward_andn: float = 3.0
    task_reward_nor: float = 4.0
    task_reward_xor: float = 4.0
    task_reward_equ: float = 5.0
    # Branch coupling: digital organisms read/produce world metabolites.
    metabolite_uptake: float = 0.0
    metabolite_deposit: float = 0.0


@dataclass(frozen=True)
class EvolutionConfig:
    generations: int
    steps_per_generation: int
    elite_fraction: float
    tournament_size: int
    extinction_min_population: int
    novelty_enabled: bool
    map_elites_enabled: bool = False
    map_elites_bins: int = 8
    novelty_k: int = 5
    novelty_archive_size: int = 256
    reseed_on_extinction: bool = True


@dataclass(frozen=True)
class LoggingConfig:
    metrics_every: int
    checkpoint_every_generations: int
    replay_every_generations: int


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int
    run_name: str
    output_dir: str
    world: WorldConfig
    embodied: EmbodiedConfig
    avida: AvidaConfig
    evolution: EvolutionConfig
    logging: LoggingConfig


@dataclass
class WorldState:
    """Shared 2D proxy-physics substrate.

    Shapes:
      terrain: [H, W, T]
      resources: [H, W, R]
      hazards: [H, W, Z]
      flow: [H, W, 2]
      curvature: [H, W, 1]
      shear: [H, W, 2]
      shear_grad: [H, W, 2]
      enrichment: [H, W, 1]
      lift: [H, W, 2]
      concentration: [H, W, C]
      metabolites: [H, W, M]
      occupancy: [H, W]
      time: scalar int
    """

    terrain: Array
    resources: Array
    hazards: Array
    flow: Array
    curvature: Array
    shear: Array
    shear_grad: Array
    enrichment: Array
    lift: Array
    concentration: Array
    metabolites: Array
    occupancy: Array
    time: Array


@dataclass
class EmbodiedPopulationState:
    positions: Array  # [N, 2] continuous in [0, 1)
    energy: Array
    age: Array
    alive: Array
    hidden: Array
    messages: Array
    genomes: dict[str, Array]
    lineage_id: Array
    parent_id: Array
    lineage_depth: Array
    last_action: Array  # [N, action_dim]
    action_history: Array  # [N, K] int discretized indices
    behavior_descriptor: Array  # [N, 2] for novelty/map-elites


@dataclass
class AvidaPopulationState:
    genomes: Array
    genome_lengths: Array
    registers: Array
    memory: Array
    ip: Array
    read_head: Array
    write_head: Array
    copied: Array
    copied_length: Array
    merit: Array
    age: Array
    alive: Array
    lineage_id: Array
    parent_id: Array
    lineage_depth: Array
    # Bitfield of logic tasks completed by each organism (NOT, NAND, AND, ...).
    # We store as an int32 with up to 9 bits set, matching the 9 reward fields.
    tasks_completed: Array | None = None
    # Last env IO input pair (per organism) used to detect logic-task outputs.
    last_input_a: Array | None = None
    last_input_b: Array | None = None


@dataclass
class SimState:
    generation: int
    step: int
    rng: PRNGKey
    world: WorldState
    embodied: EmbodiedPopulationState | None
    avida: AvidaPopulationState | None
    metrics: dict[str, Any] = field(default_factory=dict)
