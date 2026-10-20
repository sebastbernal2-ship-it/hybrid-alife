"""Bounded evolved-embodied law pilot.

This asks whether the ACTUAL evolved embodied organisms exploit an imposed law.
The law is the simulator's uniform flow field, hidden from observation but still
physically coupled through `flow_movement_scale`, and rewarded through
`flow_alignment_reward` so selection can favour alignment.

Scope is deliberately small. This is a plumbing and signal pilot, not a
five-seed held-out claim. It reports per-direction alignment and a reversed-flow
control, and it disables Avida and mutation during evaluation.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.evolution.selection import reseed_embodied, select_and_mutate_embodied
from hybrid_alife.experiments.runner import initialize_sim, step_sim
from hybrid_alife.types import (
    AvidaConfig,
    EmbodiedConfig,
    EvolutionConfig,
    ExperimentConfig,
    LoggingConfig,
    WorldConfig,
)
from hybrid_alife.world.env import positions_to_grid

HIDDEN_GROUPS = (
    "flow",
    "curvature",
    "shear",
    "shear_grad",
    "enrichment",
    "lift",
    "concentration",
    "metabolites",
)


def embodied_law_config(
    seed: int,
    *,
    generations: int = 4,
    population: int = 12,
    steps_per_generation: int = 4,
    size: int = 12,
    flow_alignment_reward: float = 0.5,
    flow_movement_scale: float = 0.05,
    visible_flow: bool = False,
    reproduce_energy_threshold: float = 22.0,
    reproduction_energy_cost: float = 5.0,
    initial_alive_fraction: float = 0.5,
) -> ExperimentConfig:
    """Tiny embodied config with the imposed law hidden but still causal."""
    if generations < 1 or population < 2 or steps_per_generation < 1:
        raise ValueError("generations, population, and steps must be positive")
    return ExperimentConfig(
        seed=seed,
        run_name=f"embodied-law-{seed}",
        output_dir="/tmp/embodied-law-pilot",
        world=WorldConfig(
            width=size,
            height=size,
            toroidal=True,
            resource_channels=1,
            hazard_channels=1,
            max_agents=population,
            max_digital_organisms=1,
            flow_noise_std=0.0,
            concentration_decay=0.99,
            concentration_diffusion=0.05,
            drifting=False,
        ),
        embodied=EmbodiedConfig(
            enabled=True,
            population_size=population,
            obs_radius=1,
            hidden_size=8,
            message_size=2,
            genome_mutation_std=0.05,
            genome_mutation_prob=0.1,
            initial_energy=20.0,
            reproduce_energy_threshold=reproduce_energy_threshold,
            reproduction_energy_cost=reproduction_energy_cost,
            initial_alive_fraction=initial_alive_fraction,
            basal_metabolic_cost=0.05,
            hidden_field_groups=() if visible_flow else HIDDEN_GROUPS,
            true_sixth_sense=True,
            flow_movement_scale=flow_movement_scale,
            flow_alignment_reward=flow_alignment_reward,
        ),
        avida=AvidaConfig(
            enabled=False,
            population_size=1,
            genome_length=8,
            max_genome_length=16,
            registers=2,
            memory_size=4,
            cycles_per_update=2,
            point_mutation_prob=0.001,
            insertion_prob=0.0,
            deletion_prob=0.0,
        ),
        evolution=EvolutionConfig(
            generations=generations,
            steps_per_generation=steps_per_generation,
            elite_fraction=0.5,
            tournament_size=2,
            extinction_min_population=2,
            novelty_enabled=False,
        ),
        logging=LoggingConfig(
            metrics_every=1,
            checkpoint_every_generations=1000,
            replay_every_generations=1000,
        ),
    )


def signed_alignment(
    moves: np.ndarray, flows: np.ndarray, alive: np.ndarray
) -> float:
    """Mean signed alignment between movement and the local law.

    Steps with no movement or no flow are excluded. Nothing is averaged across
    opposite directions here; callers report per direction.
    """
    moves = np.asarray(moves, dtype=np.float64)
    flows = np.asarray(flows, dtype=np.float64)
    alive = np.asarray(alive, dtype=bool)
    if moves.ndim != 2 or flows.ndim != 2 or moves.shape != flows.shape:
        raise ValueError("moves and flows must share a [n, 2] shape")
    move_norm = np.linalg.norm(moves, axis=-1)
    flow_norm = np.linalg.norm(flows, axis=-1)
    valid = alive & (move_norm > 1e-8) & (flow_norm > 1e-8)
    if not np.any(valid):
        return 0.0
    values = np.sum(moves[valid] * flows[valid], axis=-1) / (
        move_norm[valid] * flow_norm[valid]
    )
    return float(np.mean(values))


def _impose_flow(state, direction: tuple[float, float], magnitude: float):
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    vector = jnp.asarray(unit * magnitude, dtype=jnp.float32)
    state.world.flow = jnp.broadcast_to(vector, state.world.flow.shape)
    return state


def _local_flows(state) -> np.ndarray:
    positions = state.embodied.positions
    grid = positions_to_grid(
        positions, state.world.flow.shape[0], state.world.flow.shape[1]
    )
    return np.asarray(state.world.flow[grid[:, 0], grid[:, 1]])


def _mean_alignment_of_state(state) -> float:
    moves = np.asarray(state.embodied.last_action[:, :2])
    return signed_alignment(moves, _local_flows(state), np.asarray(state.embodied.alive))


def run_embodied_law_pilot(
    seed: int,
    *,
    directions: tuple[tuple[float, float], ...],
    magnitude: float = 0.2,
    generations: int = 4,
    population: int = 12,
    steps_per_generation: int = 4,
    size: int = 12,
) -> dict[str, Any]:
    """Evolve for a few generations, alternating the imposed law direction.

    The population therefore experiences every supplied direction, and the
    returned state can be evaluated on directions that were never used.
    """
    config = embodied_law_config(
        seed,
        generations=generations,
        population=population,
        steps_per_generation=steps_per_generation,
        size=size,
    )
    state = initialize_sim(config)
    initial_genomes = {
        name: jnp.asarray(value) for name, value in state.embodied.genomes.items()
    }
    lineage_counter = population
    training_alignment: list[float] = []
    for generation in range(generations):
        state.generation = generation
        direction = directions[generation % len(directions)]
        state = _impose_flow(state, direction, magnitude)
        for _ in range(steps_per_generation):
            state, lineage_counter = step_sim(state, config, lineage_counter)
            # The world does not drift, but re-impose so the law is exact.
            state = _impose_flow(state, direction, magnitude)
        training_alignment.append(_mean_alignment_of_state(state))
        fitness = jnp.asarray(np.asarray(state.embodied.energy))
        state.rng, k_sel = jax.random.split(state.rng)
        state.embodied = select_and_mutate_embodied(
            state.embodied, fitness, config.embodied, config.evolution, k_sel
        )
        state.rng, k_re = jax.random.split(state.rng)
        state.embodied = reseed_embodied(
            state.embodied, fitness, config.embodied, config.world, k_re
        )
    return {
        "status": "pilot_only",
        "seed": seed,
        "config": config,
        "state": state,
        "initial_genomes": initial_genomes,
        "training_alignment": training_alignment,
    }


def evaluate_frozen_alignment(
    state,
    config: ExperimentConfig,
    direction: tuple[float, float],
    *,
    magnitude: float = 0.2,
    steps: int = 12,
    genomes: dict[str, jnp.ndarray] | None = None,
) -> dict[str, float]:
    """Evaluate genomes on one direction with mutation disabled.

    ``genomes`` defaults to the evolved genomes from ``state``. Pass the initial
    genomes to measure whether evolution improved anything at all.
    """
    # Evaluate every slot. The evolved population contains revived slots whose
    # genomes changed while the original elites did not, so a half-alive target
    # would silently measure only the unchanged elites.
    eval_config = replace(
        config,
        embodied=replace(config.embodied, initial_alive_fraction=1.0),
    )
    target = initialize_sim(eval_config)
    source = genomes if genomes is not None else state.embodied.genomes
    target.embodied.genomes = {
        name: jnp.asarray(value) for name, value in source.items()
    }
    target.avida = None
    target = _impose_flow(target, direction, magnitude)
    lineage_counter = config.embodied.population_size
    alignments: list[float] = []
    for _ in range(steps):
        target, lineage_counter = step_sim(
            target, eval_config, lineage_counter, allow_reproduction=False
        )
        target = _impose_flow(target, direction, magnitude)
        alignments.append(_mean_alignment_of_state(target))
    return {
        "mean_alignment": float(np.mean(alignments)),
        "final_alignment": float(alignments[-1]) if alignments else 0.0,
    }


def per_direction_frozen_alignment(
    state,
    config: ExperimentConfig,
    directions: tuple[tuple[float, float], ...],
    *,
    magnitude: float = 0.2,
    steps: int = 12,
    genomes: dict[str, jnp.ndarray] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-direction alignment. Never antipodally averaged."""
    return {
        str(direction): evaluate_frozen_alignment(
            state,
            config,
            direction,
            magnitude=magnitude,
            steps=steps,
            genomes=genomes,
        )
        for direction in directions
    }


__all__ = [
    "HIDDEN_GROUPS",
    "embodied_law_config",
    "evaluate_frozen_alignment",
    "per_direction_frozen_alignment",
    "run_embodied_law_pilot",
    "signed_alignment",
]
