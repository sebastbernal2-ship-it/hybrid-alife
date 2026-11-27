"""Evidence for the ecosystem: windowed long-run statistics and a common-garden test."""

from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import mutate_embodied_genomes
from hybrid_alife.ecosystem import (
    DEATH_CAUSES,
    EcoParams,
    Ecosystem,
    ecosystem_step,
    initialize_ecosystem,
)
from hybrid_alife.types import ExperimentConfig
from hybrid_alife.world.compact_senses import COMPACT_SENSE_LABELS

# Genome tensors whose first non-population axis is the observation channel.
_INPUT_WEIGHTS = ("w_xz", "w_xh")


class Registry:
    """Every individual that ever lived: parent, founder, birth step, death step."""

    def __init__(self, eco: Ecosystem) -> None:
        ids = np.asarray(eco.sim.embodied.lineage_id)
        self.rows: dict[int, list] = {
            int(ids[i]): [-1, int(eco.founder[i]), 0, None, None]
            for i in np.flatnonzero(np.asarray(eco.sim.embodied.alive))
        }
        self.slot_id = ids.copy()

    def update(self, eco: Ecosystem, events) -> None:
        for slot, cause in events.deaths:
            row = self.rows[int(self.slot_id[slot])]
            row[3], row[4] = eco.step, cause
        if events.births:
            ids = np.asarray(eco.sim.embodied.lineage_id)
            parents = np.asarray(eco.sim.embodied.parent_id)
            for child, _ in events.births:
                self.rows[int(ids[child])] = [
                    int(parents[child]), int(eco.founder[child]), eco.step, None, None
                ]
            self.slot_id = ids.copy()


def _genomes_of(eco: Ecosystem, mask: np.ndarray) -> dict[str, np.ndarray]:
    return {k: np.asarray(v)[mask] for k, v in eco.sim.embodied.genomes.items()}


def save_checkpoint(path: Path, eco: Ecosystem, registry: Registry) -> None:
    sim = eco.sim
    host = replace(
        eco,
        sim=replace(
            sim,
            rng=np.asarray(sim.rng),
            world=type(sim.world)(**{k: np.asarray(v) for k, v in vars(sim.world).items()}),
            embodied=type(sim.embodied)(
                **{k: jax.tree.map(np.asarray, v) for k, v in vars(sim.embodied).items()}
            ),
        ),
    )
    path.write_bytes(pickle.dumps((host, registry)))


def load_checkpoint(path: Path) -> tuple[Ecosystem, Registry]:
    """Only for checkpoints this module wrote (pickle)."""
    eco, registry = pickle.loads(path.read_bytes())
    sim = eco.sim
    sim.rng = jnp.asarray(sim.rng)
    sim.world = type(sim.world)(**{k: jnp.asarray(v) for k, v in vars(sim.world).items()})
    sim.embodied = type(sim.embodied)(
        **{k: jax.tree.map(jnp.asarray, v) for k, v in vars(sim.embodied).items()}
    )
    return eco, registry


def long_run(
    cfg: ExperimentConfig,
    params: EcoParams,
    steps: int,
    window: int,
    out: Path,
    checkpoints: set[int],
) -> None:
    eco = initialize_ecosystem(cfg, params)
    registry = Registry(eco)
    founders = _genomes_of(eco, np.asarray(eco.sim.embodied.alive))
    windows: list[dict] = []
    counts: list[int] = []
    previous = (0, dict.fromkeys(DEATH_CAUSES, 0), 0.0, 0)
    agent_steps = 0
    for _ in range(steps):
        events = ecosystem_step(eco)
        registry.update(eco, events)
        alive = np.asarray(eco.sim.embodied.alive)
        count = int(alive.sum())
        counts.append(count)
        agent_steps += count
        if eco.step in checkpoints:
            save_checkpoint(out / f"checkpoint_{eco.step}.pkl", eco, registry)
        if eco.step % window and count:
            continue
        emb = eco.sim.embodied
        genes = {k: np.asarray(emb.genomes[k])[alive] for k in ("body_radius", "body_aspect")}
        depth = np.asarray(emb.lineage_depth)[alive]
        deaths = {c: eco.deaths[c] - previous[1][c] for c in DEATH_CAUSES}
        row = {
            "step": eco.step,
            "births": eco.births - previous[0],
            "deaths": sum(deaths.values()),
            "deaths_by_cause": deaths,
            "population_mean": float(np.mean(counts)),
            "population_min": int(min(counts)),
            "population_max": int(max(counts)),
            "population_end": count,
            "founder_lineages": len(set(eco.founder[alive].tolist())),
            "generation_depth_mean": float(depth.mean()) if count else None,
            "generation_depth_max": int(depth.max()) if count else None,
            "food_energy_per_agent_step": (eco.intake - previous[2]) / max(agent_steps, 1),
            "attack_events": eco.attacks - previous[3],
            "body_radius_mean": float(np.clip(genes["body_radius"], 0.6, 1.8).mean())
            if count else None,
            "body_aspect_mean": float(np.clip(genes["body_aspect"], 0.7, 1.4).mean())
            if count else None,
        }
        windows.append(row)
        print(json.dumps(row), flush=True)
        previous = (eco.births, dict(eco.deaths), eco.intake, eco.attacks)
        counts, agent_steps = [], 0
        summary = {
            "seed": cfg.seed,
            "steps_run": eco.step,
            "extinct": count == 0,
            "founders": int(len(founders["body_radius"])),
            "births_total": eco.births,
            "deaths_total": dict(eco.deaths),
            "windows": windows,
        }
        (out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
        if count:
            payload = {
                "cfg": cfg,
                "params": params,
                "founders": founders,
                "evolved": _genomes_of(eco, alive),
                "evolved_depth": depth,
            }
            (out / "genomes.pkl").write_bytes(pickle.dumps(payload))
        else:
            break


def _resample(genomes: dict, count: int, rng: np.random.Generator) -> dict:
    total = len(genomes["body_radius"])
    picks = rng.choice(total, size=count, replace=count > total)
    return {k: v[picks] for k, v in genomes.items()}


def _blind_to(genomes: dict, prefixes: tuple[str, ...]) -> dict:
    """Zero the input weights of the channels starting with any of ``prefixes``."""
    channels = [i for i, name in enumerate(COMPACT_SENSE_LABELS) if name.startswith(prefixes)]
    out = {k: np.array(v) for k, v in genomes.items()}
    for name in _INPUT_WEIGHTS:
        out[name][:, channels, :] = 0.0
    return out


def garden_trial(
    cfg: ExperimentConfig, params: EcoParams, genomes: dict, steps: int
) -> dict[str, float]:
    """Frozen genomes (no reproduction) in a fresh world fixed by ``cfg.seed``."""
    eco = initialize_ecosystem(cfg, params)
    n = cfg.embodied.population_size
    count = len(genomes["body_radius"])
    emb = eco.sim.embodied
    for key, value in genomes.items():
        emb.genomes[key] = emb.genomes[key].at[:count].set(jnp.asarray(value))
    alive = jnp.arange(n) < count
    emb.alive = alive
    emb.energy = jnp.where(alive, cfg.embodied.initial_energy, 0.0).astype(jnp.float32)
    eaten = 0.0
    agent_steps = 0
    toward: list[float] = []
    hazard = 0.0
    for _ in range(steps):
        live = np.asarray(eco.sim.embodied.alive)
        old = np.asarray(eco.sim.embodied.positions)
        events = ecosystem_step(eco, allow_reproduction=False)
        eaten += float(events.eaters.sum())
        agent_steps += int(live.sum())
        food = events.observation[live, 0:2]
        new = np.asarray(eco.sim.embodied.positions)
        move = (new - old)[live]
        move -= np.round(move)  # shortest displacement on the torus
        hazard += float((events.observation[live, 8] > 0.0).sum())
        norm = np.linalg.norm(food, axis=1) * np.linalg.norm(move, axis=1)
        valid = norm > 1e-8
        toward.extend((np.sum(food[valid] * move[valid], axis=1) / norm[valid]).tolist())
    return {
        "food_energy_per_founder": eaten / count,
        "food_energy_per_agent_step": eaten / max(agent_steps, 1),
        "survival_fraction": float(np.asarray(eco.sim.embodied.alive).sum()) / count,
        "share_of_steps_in_hazard": hazard / max(agent_steps, 1),
        "steering_toward_food": float(np.mean(toward)) if toward else 0.0,
    }


def common_garden(saved: dict, *, eval_seeds: int, steps: int, cohort: int) -> dict:
    """Founders vs evolved genomes, plus two controls, in paired fresh worlds.

    ``mutated_only``: founder genomes mutated as many times as the evolved median
    generation depth, with no selection (a depth-matched control, not a replay of each
    lineage). ``evolved_food_blind``: evolved genomes with the three food input
    channels disconnected. ``evolved_all_blind``: all nine external channels
    disconnected (the energy sense stays).
    """
    cfg, params = saved["cfg"], saved["params"]
    rng = np.random.default_rng(12345 + cfg.seed)
    founders = _resample(saved["founders"], cohort, rng)
    evolved = _resample(saved["evolved"], cohort, rng)
    drifted = {k: jnp.asarray(v) for k, v in founders.items()}
    generations = int(np.median(saved["evolved_depth"]))
    key = jax.random.PRNGKey(cfg.seed + 99)
    mutate = jax.jit(lambda g, k: mutate_embodied_genomes(g, cfg.embodied, k))
    for _ in range(generations):
        key, sub = jax.random.split(key)
        drifted = mutate(drifted, sub)
    cohorts = {
        "founders": founders,
        "mutated_only": {k: np.asarray(v) for k, v in drifted.items()},
        "evolved": evolved,
        "evolved_food_blind": _blind_to(evolved, ("food",)),
        "evolved_all_blind": _blind_to(evolved, ("food", "other", "hazard")),
    }
    rows: dict[str, dict[str, list[float]]] = {name: {} for name in cohorts}
    worlds = [1000 + i for i in range(eval_seeds)]
    for world_seed in worlds:
        trial_cfg = replace(cfg, seed=world_seed)
        for name, genomes in cohorts.items():
            for metric, value in garden_trial(trial_cfg, params, genomes, steps).items():
                rows[name].setdefault(metric, []).append(value)
    return {
        "train_seed": cfg.seed,
        "eval_world_seeds": worlds,
        "steps": steps,
        "cohort": cohort,
        "mutated_only_generations": generations,
        "cohorts": rows,
    }
