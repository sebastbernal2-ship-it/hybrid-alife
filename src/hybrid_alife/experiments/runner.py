"""Experiment runner for hybrid-alife.

Wires the world, embodied branch, Avida branch, evolutionary loop, metrics,
checkpoints, and archives into one coherent generation loop.
"""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import yaml
from rich.console import Console

from hybrid_alife.agents.avida_vm import initialize_avida_population, step_avida_population
from hybrid_alife.agents.embodied import (
    act_embodied,
    apply_embodied_actions,
    apply_reproduction,
    initialize_embodied_population,
    observe_embodied,
)
from hybrid_alife.evolution.archives import MapElitesArchive, NoveltyArchive
from hybrid_alife.evolution.selection import (
    reseed_avida,
    reseed_embodied,
    select_and_mutate_embodied,
)
from hybrid_alife.logging.jsonl import JsonlWriter
from hybrid_alife.metrics.core import collect_full_metrics
from hybrid_alife.metrics.qd import archive_entropy, coverage as qd_coverage, qd_score
from hybrid_alife.replay.checkpoint import save_checkpoint
from hybrid_alife.types import (
    AvidaConfig,
    EmbodiedConfig,
    EvolutionConfig,
    ExperimentConfig,
    LoggingConfig,
    SimState,
    WorldConfig,
)
from hybrid_alife.world.env import compute_occupancy, step_world
from hybrid_alife.world.env import initialize_world

console = Console()


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


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
    field_info = {f.name: f for f in fields(cls)}
    kwargs = {}
    for name, fld in field_info.items():
        if name in raw:
            kwargs[name] = raw[name]
        elif fld.default is not fld.default_factory and fld.default is not getattr(fld, "MISSING", None):  # type: ignore[attr-defined]
            # leave to default
            pass
    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Sim init & step
# ---------------------------------------------------------------------------


def initialize_sim(cfg: ExperimentConfig) -> SimState:
    key = jax.random.PRNGKey(cfg.seed)
    k_world, k_embodied, k_avida, k_next = jax.random.split(key, 4)
    world = initialize_world(cfg.world, k_world)
    embodied = (
        initialize_embodied_population(cfg.embodied, cfg.world, k_embodied)
        if cfg.embodied.enabled
        else None
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


def step_sim(state: SimState, cfg: ExperimentConfig, lineage_counter: int) -> tuple[SimState, int]:
    """Single integration step. Returns (state, new lineage_counter)."""
    state.rng, k_obs, k_ctrl, k_act, k_repro, k_avida = jax.random.split(state.rng, 6)
    state = step_world(state, cfg.world)

    if state.embodied is not None and cfg.embodied.enabled:
        obs = observe_embodied(state.embodied, state.world, cfg.embodied, cfg.world, k_obs)
        actions, embodied = act_embodied(state.embodied, obs, cfg.embodied, k_ctrl)
        embodied, new_world, repro_gate = apply_embodied_actions(
            embodied, actions, state.world, cfg.embodied, cfg.world, k_act
        )
        state.world = new_world
        # Reproduction
        embodied = apply_reproduction(
            embodied, repro_gate, cfg.embodied, lineage_counter, k_repro
        )
        lineage_counter += embodied.alive.shape[0]
        state.embodied = embodied
        # Update occupancy on world from new positions for crowd metrics
        state.world.occupancy = compute_occupancy(
            embodied.positions, embodied.alive, cfg.world.height, cfg.world.width
        )

    if state.avida is not None and cfg.avida.enabled:
        # Map each digital organism onto a world cell. When embodied agents
        # exist we wrap their positions around (so each Avida slot picks up
        # an embodied agent's local metabolite/concentration values, which
        # is what couples the two branches). Otherwise fall back to a
        # deterministic embedding so env IO and tasks are still well-defined.
        n_av = state.avida.alive.shape[0]
        if state.embodied is not None and state.embodied.positions.shape[0] > 0:
            n_em = state.embodied.positions.shape[0]
            wrap_idx = jnp.arange(n_av) % n_em
            avida_positions = state.embodied.positions[wrap_idx]
        else:
            grid = jnp.linspace(0.05, 0.95, n_av)
            avida_positions = jnp.stack([grid, grid[::-1]], axis=-1)
        state.avida, lineage_counter = step_avida_population(
            state.avida, state.world, cfg.avida, lineage_counter, k_avida, avida_positions
        )

    state.step += 1
    return state, lineage_counter


# ---------------------------------------------------------------------------
# Full experiment loop
# ---------------------------------------------------------------------------


def run_experiment(cfg: ExperimentConfig) -> SimState:
    output_dir = Path(cfg.output_dir) / cfg.run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    raw_config = _serialize_config(cfg)
    (output_dir / "config.yaml").write_text(yaml.safe_dump(raw_config), encoding="utf-8")

    state = initialize_sim(cfg)
    lineage_counter = (
        (cfg.embodied.population_size if cfg.embodied.enabled else 0)
        + (cfg.avida.population_size if cfg.avida.enabled else 0)
    )

    novelty = NoveltyArchive(capacity=cfg.evolution.novelty_archive_size, k=cfg.evolution.novelty_k)
    map_elites = MapElitesArchive(bins=cfg.evolution.map_elites_bins)

    console.print(f"[bold]Starting run[/bold] {cfg.run_name} -> {output_dir}")

    with JsonlWriter(metrics_path) as writer:
        for generation in range(cfg.evolution.generations):
            state.generation = generation

            for _ in range(cfg.evolution.steps_per_generation):
                state, lineage_counter = step_sim(state, cfg, lineage_counter)
                if state.step % cfg.logging.metrics_every == 0:
                    metrics = collect_full_metrics(state)
                    state.metrics = metrics
                    writer.write(metrics)

            # End-of-generation: archives, selection, reseeding
            if state.embodied is not None:
                bd = np.asarray(state.embodied.behavior_descriptor)
                alive_np = np.asarray(state.embodied.alive)
                alive_bd = bd[alive_np]
                # Use energy as fitness proxy
                fitness_np = np.asarray(state.embodied.energy)
                novelty.add_batch(alive_bd)
                if cfg.evolution.map_elites_enabled:
                    map_elites.update(alive_bd, fitness_np[alive_np])

                # Selection: tournament + mutation revives dead
                state.rng, k_sel = jax.random.split(state.rng)
                import jax.numpy as jnp

                fitness = jnp.asarray(fitness_np)
                state.embodied = select_and_mutate_embodied(
                    state.embodied, fitness, cfg.embodied, cfg.evolution, k_sel
                )
                # Extinction reseeding
                state.rng, k_re = jax.random.split(state.rng)
                if cfg.evolution.reseed_on_extinction:
                    state.embodied = reseed_embodied(
                        state.embodied, fitness, cfg.embodied, cfg.world, k_re
                    )

            if state.avida is not None and cfg.evolution.reseed_on_extinction:
                state.rng, k_av_re = jax.random.split(state.rng)
                state.avida = reseed_avida(state.avida, fraction=0.1, key=k_av_re)

            # Per-generation QD summary (logged even when map_elites disabled
            # so downstream tooling always sees the keys).
            gen_record = {
                "kind": "generation",
                "generation": generation,
                "step": state.step,
                "novelty_archive_size": int(novelty.descriptors.shape[0]),
                "map_elites_coverage": qd_coverage(map_elites),
                "qd_score": qd_score(map_elites),
                "archive_entropy": archive_entropy(map_elites),
            }
            writer.write(gen_record)

            if (generation + 1) % cfg.logging.checkpoint_every_generations == 0:
                save_checkpoint(
                    output_dir / f"checkpoint_gen{generation:05d}.pkl", state, raw_config
                )

            console.print(
                {
                    "generation": generation,
                    "step": state.step,
                    "novelty_archive_size": int(novelty.descriptors.shape[0]),
                    "map_elites_coverage": gen_record["map_elites_coverage"],
                    "qd_score": gen_record["qd_score"],
                    "archive_entropy": gen_record["archive_entropy"],
                    **state.metrics,
                }
            )

    # Final checkpoint and final archives
    save_checkpoint(output_dir / "checkpoint_final.pkl", state, raw_config)
    np.savez(
        output_dir / "novelty_archive.npz",
        descriptors=novelty.descriptors,
    )
    np.savez(
        output_dir / "map_elites.npz",
        fitness=map_elites.grid_fitness,
        filled=map_elites.grid_filled,
        descriptor=map_elites.grid_descriptor,
    )
    return state


def _serialize_config(cfg: ExperimentConfig) -> dict[str, Any]:
    return {
        "seed": cfg.seed,
        "run_name": cfg.run_name,
        "output_dir": cfg.output_dir,
        "world": {f.name: getattr(cfg.world, f.name) for f in fields(WorldConfig)},
        "embodied": {f.name: getattr(cfg.embodied, f.name) for f in fields(EmbodiedConfig)},
        "avida": {f.name: getattr(cfg.avida, f.name) for f in fields(AvidaConfig)},
        "evolution": {f.name: getattr(cfg.evolution, f.name) for f in fields(EvolutionConfig)},
        "logging": {f.name: getattr(cfg.logging, f.name) for f in fields(LoggingConfig)},
    }
