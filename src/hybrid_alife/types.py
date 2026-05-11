"""Core dataclasses and typed state containers for hybrid-alife."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import jax
import jax.numpy as jnp

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
      enrichment: [H, W, 1]
      lift: [H, W, 2]
      concentration: [H, W, C]
      occupancy: [H, W]
    """

    terrain: Array
    resources: Array
    hazards: Array
    flow: Array
    curvature: Array
    shear: Array
    enrichment: Array
    lift: Array
    concentration: Array
    occupancy: Array


@dataclass
class EmbodiedPopulationState:
    positions: Array
    energy: Array
    age: Array
    alive: Array
    hidden: Array
    messages: Array
    genomes: dict[str, Array]
    lineage_id: Array
    parent_id: Array


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
    merit: Array
    age: Array
    alive: Array
    lineage_id: Array
    parent_id: Array


@dataclass
class SimState:
    generation: int
    step: int
    rng: PRNGKey
    world: WorldState
    embodied: EmbodiedPopulationState | None
    avida: AvidaPopulationState | None
    metrics: dict[str, Any]

