"""Experiment runner for hybrid-alife."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import yaml
from rich.console import Console

from hybrid_alife.agents.avida_vm import initialize_avida_population, step_avida_population
from hybrid_alife.agents.embodied import (
    act_embodied,
    initialize_embodied_population,
    observe_embodied,
)
from hybrid_alife.types import (
    AvidaConfig,
    EmbodiedConfig,
    EvolutionConfig,
    ExperimentConfig,
    LoggingConfig,
    SimState,
    WorldConfig,
)
from hybrid_alife.world.env import initialize_world, step_world

console = Console()


def load_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig(
        seed=raw["seed"],
        run_name=raw["run_name"],
        output_dir=raw["output_dir"],
        world=_coerce_dataclass(WorldConfig, raw["world"]),
        embodied=_coerce_dataclass(EmbodiedConfig, raw["embodied"]),
        avida=_coerce_dataclass(AvidaConfig, raw["avida"]),
        evolution=_coerce_dataclass(EvolutionConfig, raw["evolution"]),
        logging=_coerce_dataclass(LoggingConfig, raw["logging"]),
    )


def _coerce_dataclass(cls: type, raw: dict[str, Any]) -> Any:
    names = {field.name for field in fields(cls)}
    return cls(**{name: raw[name] for name in names})


def initialize_sim(cfg: ExperimentConfig) -> SimState:
    key = jax.random.PRNGKey(cfg.seed)
    k_world, k_embodied, k_avida, k_next = jax.random.split(key, 4)
    world = initialize_world(cfg.world, k_world)
    embodied = (
        initialize_embodied_population(cfg.embodied, k_embodied) if cfg.embodied.enabled else None
    )
    avida = initialize_avida_population(cfg.avida, k_avida) if cfg.avida.enabled else None
    return SimState(
        generation=0,
        step=0,
        rng=k_next,
        world=world,
        embodied=embodied,
        avida=avida,
        metrics={},
    )


def run_experiment(cfg: ExperimentConfig) -> SimState:
    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    state = initialize_sim(cfg)
    console.print(f"[bold]Starting run[/bold] {cfg.run_name}")

    for generation in range(cfg.evolution.generations):
        state.generation = generation
        for _ in range(cfg.evolution.steps_per_generation):
            state = step_sim(state, cfg)
            if state.step % cfg.logging.metrics_every == 0:
                state.metrics = collect_smoke_metrics(state)
        console.print(
            {
                "generation": generation,
                "step": state.step,
                **state.metrics,
            }
        )
    return state


def step_sim(state: SimState, cfg: ExperimentConfig) -> SimState:
    state.rng, k_avida = jax.random.split(state.rng)
    state = step_world(state, cfg.world)

    if state.embodied is not None:
        obs = observe_embodied(state.embodied, state.world, cfg.embodied)
        _actions, embodied = act_embodied(state.embodied, obs, cfg.embodied)
        embodied.age = embodied.age + embodied.alive.astype("int32")
        embodied.energy = embodied.energy - cfg.embodied.basal_metabolic_cost
        state.embodied = embodied

    if state.avida is not None:
        state.avida = step_avida_population(state.avida, state.world, cfg.avida, k_avida)

    state.step += 1
    return state


def collect_smoke_metrics(state: SimState) -> dict[str, float | int]:
    embodied_alive = (
        int(state.embodied.alive.sum()) if state.embodied is not None else 0
    )
    avida_alive = int(state.avida.alive.sum()) if state.avida is not None else 0
    return {
        "embodied_alive": embodied_alive,
        "avida_alive": avida_alive,
        "mean_enrichment": float(state.world.enrichment.mean()),
        "mean_concentration": float(state.world.concentration.mean()),
    }

