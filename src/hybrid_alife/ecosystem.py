"""Opt-in continuous ecosystem on top of the embodied simulation.

No generations, tournament selection, elite copying or reseeding: after the founders,
the only way to appear is to be born from a living parent (asexual, mutated copy) and
the only way to persist is not to die. A population that dies out stays dead.

The physics and controller are the core ones (``step_world``, ``observe_embodied``,
``act_embodied``, ``apply_embodied_actions``, ``apply_reproduction``). This layer adds
only an environment: discrete food pellets that sprout in slowly drifting patches, and
slowly drifting toxic patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from hybrid_alife.agents.embodied import (
    act_embodied,
    apply_embodied_actions,
    apply_reproduction,
    observe_embodied,
)
from hybrid_alife.experiments.runner import initialize_sim, load_config
from hybrid_alife.types import (
    EmbodiedPopulationState,
    ExperimentConfig,
    SimState,
    WorldState,
)
from hybrid_alife.world.env import step_world

DEATH_CAUSES = ("starved", "old_age", "hazard", "attacked")


@dataclass(frozen=True)
class EcoParams:
    food_per_step: float = 10.0
    initial_food: int = 900  # pellets sprouted before step 0
    food_patches: int = 7
    food_patch_sigma: float = 0.07
    food_background: float = 0.15  # share of pellets sprouting anywhere
    patch_drift: float = 0.0006  # world fraction per step
    hazard_patches: int = 4
    hazard_radius: float = 0.06
    hazard_strength: float = 0.6  # energy per step at a patch centre
    hazard_drift: float = 0.0002
    pellet_floor: float = 0.15  # smaller leftovers vanish
    food_rot: float = 0.0  # chance per step that an uneaten pellet rots away
    # The world gets harsher on a fixed schedule that ignores the population: rot is 0
    # until rot_start, then rises linearly to food_rot over rot_ramp steps.
    rot_start: int = 0
    rot_ramp: int = 0


@dataclass
class Ecosystem:
    cfg: ExperimentConfig
    params: EcoParams
    sim: SimState
    rng: np.random.Generator
    patch_pos: np.ndarray
    patch_vel: np.ndarray
    hazard_pos: np.ndarray
    hazard_vel: np.ndarray
    next_id: int
    founder: np.ndarray  # [N] founder slot index per living individual
    step: int = 0
    births: int = 0
    deaths: dict[str, int] = field(default_factory=lambda: dict.fromkeys(DEATH_CAUSES, 0))
    attacks: int = 0
    intake: float = 0.0


@dataclass
class StepEvents:
    """What happened in one step, in slot indices. All of it comes from the simulator."""

    births: list[tuple[int, int]]  # (child slot, parent slot)
    deaths: list[tuple[int, str]]  # (slot, cause)
    attacks: list[tuple[int, int]]  # (attacker slot, target slot)
    eaters: np.ndarray  # [N] food energy eaten this step
    observation: np.ndarray  # [N, obs]
    actions: np.ndarray  # [N, act]


def load_ecosystem_config(path: str | Path) -> tuple[ExperimentConfig, EcoParams]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return load_config(path), EcoParams(**raw.get("ecosystem", {}))


def _blobs(h: int, w: int, centres: np.ndarray, sigma: float) -> np.ndarray:
    rows = (np.arange(h) + 0.5) / h
    cols = (np.arange(w) + 0.5) / w
    dy = rows[None, :] - centres[:, 0, None]
    dx = cols[None, :] - centres[:, 1, None]
    dy -= np.round(dy)
    dx -= np.round(dx)
    gy = np.exp(-0.5 * (dy / sigma) ** 2)
    gx = np.exp(-0.5 * (dx / sigma) ** 2)
    return np.einsum("ky,kx->yx", gy, gx)


def _hazard_field(eco_params: EcoParams, h: int, w: int, centres: np.ndarray) -> np.ndarray:
    if len(centres) == 0:
        return np.zeros((h, w), np.float32)
    blob = np.minimum(_blobs(h, w, centres, eco_params.hazard_radius / 2.0), 1.0)
    return (eco_params.hazard_strength * np.where(blob > 0.2, blob, 0.0)).astype(np.float32)


def _sprout(eco: Ecosystem, resources: np.ndarray, count: int) -> None:
    h, w = resources.shape
    p = eco.params
    density = _blobs(h, w, eco.patch_pos, p.food_patch_sigma).ravel()
    density = (1.0 - p.food_background) * density / density.sum() + p.food_background / (h * w)
    cells = eco.rng.choice(h * w, size=count, p=density / density.sum())
    resources.ravel()[cells] = 1.0


def initialize_ecosystem(
    cfg: ExperimentConfig, params: EcoParams
) -> Ecosystem:
    if not cfg.embodied.compact_senses or cfg.avida.enabled:
        raise ValueError("the ecosystem needs compact_senses and no Avida branch")
    sim = initialize_sim(cfg)
    rng = np.random.default_rng(cfg.seed + 7919)

    def drift(count: int, speed: float) -> np.ndarray:
        angle = rng.uniform(0.0, 2.0 * np.pi, count)
        return speed * np.stack([np.sin(angle), np.cos(angle)], axis=-1)

    n = cfg.embodied.population_size
    eco = Ecosystem(
        cfg=cfg,
        params=params,
        sim=sim,
        rng=rng,
        patch_pos=rng.uniform(size=(params.food_patches, 2)),
        patch_vel=drift(params.food_patches, params.patch_drift),
        hazard_pos=rng.uniform(size=(params.hazard_patches, 2)),
        hazard_vel=drift(params.hazard_patches, params.hazard_drift),
        next_id=n,
        founder=np.arange(n),
    )
    h, w = cfg.world.height, cfg.world.width
    resources = np.zeros((h, w), np.float32)
    if params.initial_food:
        _sprout(eco, resources, params.initial_food)
    eco.sim.world = replace(
        eco.sim.world,
        resources=jnp.asarray(resources)[..., None],
        hazards=jnp.asarray(_hazard_field(params, h, w, eco.hazard_pos))[..., None],
    )
    return eco


def _pop_to_dict(pop: EmbodiedPopulationState) -> dict:
    return {k: v for k, v in vars(pop).items()}


@partial(jax.jit, static_argnums=(0,))
def _physics(cfg: ExperimentConfig, pop: dict, world: dict, rng: jax.Array):
    """``runner.step_sim`` up to, but not including, reproduction, compiled."""
    rng, k_obs, k_ctrl, k_act, k_repro, _ = jax.random.split(rng, 6)
    sim = SimState(0, 0, rng, WorldState(**world), None, None)
    sim = step_world(sim, cfg.world)
    embodied = EmbodiedPopulationState(**pop)
    obs = observe_embodied(embodied, sim.world, cfg.embodied, cfg.world, k_obs)
    actions, embodied = act_embodied(embodied, obs, cfg.embodied, k_ctrl)
    embodied, new_world, gate = apply_embodied_actions(
        embodied, actions, sim.world, cfg.embodied, cfg.world, k_act
    )
    return _pop_to_dict(embodied), vars(new_world), sim.rng, obs, actions, gate, k_repro


def ecosystem_step(eco: Ecosystem, *, allow_reproduction: bool = True) -> StepEvents:
    cfg, p = eco.cfg, eco.params
    emb_cfg = cfg.embodied
    before = eco.sim.embodied
    alive_before = np.asarray(before.alive)
    pop, world, rng, obs, actions, gate, k_repro = _physics(
        cfg, _pop_to_dict(before), dict(vars(eco.sim.world)), eco.sim.rng
    )
    eco.sim.rng = rng
    embodied = EmbodiedPopulationState(**pop)
    new_world = WorldState(**world)

    alive_mid = np.asarray(embodied.alive)
    positions = np.asarray(embodied.positions)
    h, w = cfg.world.height, cfg.world.width
    hazard = np.asarray(new_world.hazards[..., 0])
    attacks: list[tuple[int, int]] = []
    if embodied.last_attack is not None:
        mask = np.asarray(embodied.last_attack)
        targets = np.asarray(embodied.last_attack_target)
        attacks = [(int(i), int(targets[i])) for i in np.flatnonzero(mask)]
    attacked = {t for _, t in attacks}
    ages = np.asarray(before.age)
    deaths: list[tuple[int, str]] = []
    for i in np.flatnonzero(alive_before & ~alive_mid):
        row = min(int(positions[i, 0] * h), h - 1)
        col = min(int(positions[i, 1] * w), w - 1)
        if int(i) in attacked:
            cause = "attacked"
        elif hazard[row, col] > 0.0:
            cause = "hazard"
        elif emb_cfg.senescence_cost > 0.0 and ages[i] > emb_cfg.senescence_age:
            cause = "old_age"
        else:
            cause = "starved"
        deaths.append((int(i), cause))
        eco.deaths[cause] += 1
    eaters = np.asarray(embodied.last_intake)

    births: list[tuple[int, int]] = []
    if allow_reproduction:
        gate_np = np.asarray(gate)
        energy = np.asarray(embodied.energy)
        eligible = (
            alive_mid
            & (energy >= emb_cfg.reproduce_energy_threshold)
            & (gate_np > emb_cfg.repro_gate_threshold)
        )
        if eligible.any() and not alive_mid.all():
            old_ids = np.asarray(embodied.lineage_id)
            embodied, count = apply_reproduction(embodied, gate, emb_cfg, eco.next_id, k_repro)
            eco.next_id += emb_cfg.population_size
            new_ids = np.asarray(embodied.lineage_id)
            parent_ids = np.asarray(embodied.parent_id)
            slot_of = {int(old_ids[i]): int(i) for i in np.flatnonzero(alive_mid)}
            founder = eco.founder.copy()
            for child in np.flatnonzero(new_ids != old_ids):
                parent = slot_of[int(parent_ids[child])]
                births.append((int(child), parent))
                founder[child] = eco.founder[parent]
            if len(births) != count:
                raise RuntimeError("birth bookkeeping disagrees with the simulator")
            eco.founder = founder
            eco.births += count

    # Environment: remove crumbs, drift the patches, sprout new pellets.
    resources = np.array(new_world.resources[..., 0])
    resources[resources < p.pellet_floor] = 0.0
    rot = p.food_rot * float(np.clip((eco.step - p.rot_start) / max(p.rot_ramp, 1), 0.0, 1.0))
    if rot > 0.0:
        resources[eco.rng.random(resources.shape) < rot] = 0.0
    eco.patch_pos = (eco.patch_pos + eco.patch_vel) % 1.0
    eco.hazard_pos = (eco.hazard_pos + eco.hazard_vel) % 1.0
    sprouts = int(eco.rng.poisson(p.food_per_step))
    if sprouts:
        _sprout(eco, resources, sprouts)
    eco.sim.world = replace(
        new_world,
        resources=jnp.asarray(resources)[..., None],
        hazards=jnp.asarray(_hazard_field(p, h, w, eco.hazard_pos))[..., None],
    )
    eco.sim.embodied = embodied
    eco.step += 1
    eco.sim.step = eco.step
    eco.attacks += len(attacks)
    eco.intake += float(eaters.sum())
    return StepEvents(births, deaths, attacks, eaters, np.asarray(obs), np.asarray(actions))
