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


@dataclass
class SimState:
    generation: int
    step: int
    rng: PRNGKey
    world: WorldState
    embodied: EmbodiedPopulationState | None
    avida: AvidaPopulationState | None
    metrics: dict[str, Any] = field(default_factory=dict)
