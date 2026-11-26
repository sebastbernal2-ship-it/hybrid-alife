"""Run the live LBM + embodied simulation and record it for the browser viewer.

Display frame: screen x is world axis 0 (the channel's flow direction), screen y (pointing
up) is world axis 1. Field arrays are recorded as [y, x]; vectors as (x, y). Velocities are
in lattice units (cells per LBM step); one simulation step is ``lbm_substeps`` LBM steps.
Recorded quantities come from simulator state. Two are derived: ``swim`` is the commanded
displacement (movement action times the 0.05 maximum step) and ``founder`` follows parent ids.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, step_sim
from hybrid_alife.physics import lbm
from hybrid_alife.types import (
    AvidaConfig,
    EmbodiedConfig,
    EvolutionConfig,
    ExperimentConfig,
    LoggingConfig,
    WorldConfig,
)
from hybrid_alife.viewer.replay import write_replay

U_SCALE = 3e-5  # lattice units per int16 count
SCENES = ("obstacle", "periodic")


def viewer_config(
    scene: str = "obstacle", grid: int = 256, population: int = 40, seed: int = 3,
    substeps: int = 60, drag: float = 0.007,
) -> ExperimentConfig:
    if scene not in SCENES:
        raise ValueError(f"scene must be one of {SCENES}, got {scene!r}")
    peak = 0.08
    world = WorldConfig(
        width=grid, height=grid, toroidal=True, resource_channels=3, hazard_channels=2,
        max_agents=population, max_digital_organisms=0, flow_noise_std=0.0,
        concentration_decay=0.955, concentration_diffusion=0.22,
        # Semi-Lagrangian step in cells = strength * flow, so this value advects the signal
        # exactly as far per sim step as the fluid moves (lbm_substeps lattice velocities).
        flow_advection_strength=peak * substeps, flow_solver="lbm_live",
        curvature_regime="obstacle" if scene == "obstacle" else "straight",
        lbm_viscosity=0.008, lbm_peak_speed=peak, lbm_substeps=substeps,
        lbm_agent_drag=drag, lbm_agent_coupling="dipole", lbm_dipole_length=6.0,
        lbm_box="channel" if scene == "obstacle" else "periodic",
        lbm_law_relax_steps=0.0 if scene == "obstacle" else 2000.0,
    )
    embodied = EmbodiedConfig(
        enabled=True, population_size=population, obs_radius=2, hidden_size=12,
        message_size=4, genome_mutation_std=0.05, genome_mutation_prob=0.1,
        initial_energy=24.0, reproduce_energy_threshold=27.0,
        reproduction_energy_cost=10.0, basal_metabolic_cost=0.05,
        body_morphology=True, chemical_signaling=True, signal_deposit=1.0,
        stochastic_actions=True, action_temperature=0.5,
        # Passive drift equals tracer drift: flow * scale = u * substeps / grid.
        flow_movement_scale=peak * substeps / grid, max_energy=60.0,
        initial_alive_fraction=0.45,
    )
    return ExperimentConfig(
        seed=seed, run_name=f"viewer_{scene}", output_dir="outputs/viewer",
        world=world, embodied=embodied,
        avida=AvidaConfig(False, 1, 8, 16, 4, 8, 1, 0.0, 0.0, 0.0),
        evolution=EvolutionConfig(1, 1, 0.1, 3, 2, False),
        logging=LoggingConfig(1, 1, 1),
    )


def obstacle_channel(cfg: WorldConfig, warm_steps: int) -> lbm.LiveLBM:
    """Channel with a small cylinder, started from a parabolic profile plus a weak
    transverse perturbation (to break symmetry) and relaxed with the live LBM kernel.

    The stock ``new_live_lbm`` starts from rest with a wider cylinder; at this resolution
    that takes far too long to develop a wake. Same kernel, different initial condition.
    """
    n, peak = cfg.width, cfg.lbm_peak_speed
    mask = lbm.channel_mask(n, n, n / 12)  # [y, x]; cylinder at x = n/4
    y = np.arange(n)
    profile = peak * np.clip(4 * (y - 0.5) * (n - 1.5 - y) / (n - 2) ** 2, 0, None)
    # The kernel's arrays are [y, x]: world axis 1 first, world axis 0 (streamwise) second.
    u = np.ascontiguousarray(np.tile(profile[:, None], (1, n)), np.float64)
    v = np.tile(0.05 * peak * np.sin(4 * np.pi * y / n), (n, 1))
    u[mask > 0] = 0.0
    v[mask > 0] = 0.0
    f = np.zeros((n, n, 9))
    lbm._load().lbm_init(n, n, 1.0, u, np.ascontiguousarray(v), f)
    fluid = lbm.LiveLBM(
        f, mask, cfg.lbm_viscosity, lbm.channel_body_force(n, cfg.lbm_viscosity, peak),
        peak, u, v, np.ones((n, n)), np.zeros((n, n, 2)),
    )
    return fluid.advance(warm_steps)


def record(
    cfg: ExperimentConfig, scene: str, steps: int = 240, field_every: int = 3,
    warm_steps: int = 30000,
) -> tuple[dict, dict[str, tuple[np.ndarray, str]]]:
    if steps < 1 or field_every < 1 or steps % field_every:
        raise ValueError("need steps >= 1, field_every >= 1, steps divisible by field_every")
    state = initialize_sim(cfg)
    if scene == "obstacle":
        state.lbm = obstacle_channel(cfg.world, warm_steps)
    else:
        state.lbm = lbm.LiveLBM.uniform(
            cfg.world.height, cfg.world.width, nu=cfg.world.lbm_viscosity,
            peak_speed=cfg.world.lbm_peak_speed, velocity=(0.04, 0.0),
            relax_steps=cfg.world.lbm_law_relax_steps,
        )
    state.world = replace(
        state.world,
        flow=state.world.flow.at[...].set(state.lbm.flow()),
        concentration=state.world.concentration * 0.0,
    )
    counter = cfg.embodied.population_size
    founder: dict[int, int] = {}
    fields: list[np.ndarray] = []
    signals: list[np.ndarray] = []
    rows: dict[str, list[np.ndarray]] = {}

    def snapshot() -> None:
        pop = state.embodied
        alive = np.asarray(pop.alive)
        ids, parents = np.asarray(pop.lineage_id), np.asarray(pop.parent_id)
        for i in np.flatnonzero(alive):
            lid = int(ids[i])
            if lid not in founder:
                founder[lid] = founder.get(int(parents[i]), lid)
        action = np.asarray(pop.last_action, np.float32)
        values = {
            "position": np.asarray(pop.positions, np.float32),
            # Commanded swim displacement, fraction of domain per sim step.
            "swim": action[:, :2] * 0.05,
            "alive": alive.astype(np.uint8),
            "energy": np.asarray(pop.energy, np.float32),
            "radius": np.clip(np.asarray(pop.genomes["body_radius"], np.float32), 0.6, 1.8),
            "aspect": np.asarray(pop.genomes["body_aspect"], np.float32),
            "lineage": ids.astype(np.int32),
            "parent": parents.astype(np.int32),
            "founder": np.array(
                [founder.get(int(v), -1) for v in ids], np.int32
            ),
            "depth": np.asarray(pop.lineage_depth, np.int32),
            "age": np.asarray(pop.age, np.int32),
            "emitting": (alive & (action[:, 7] > cfg.embodied.message_gate_threshold)
                         & (np.asarray(pop.age) > 0)).astype(np.uint8),
        }
        for key, value in values.items():
            rows.setdefault(key, []).append(value.copy())

    def snapshot_fields() -> None:
        uv = state.lbm.velocity().transpose(1, 0, 2)  # -> [y, x, (ux, uy)]
        fields.append(np.round(np.clip(uv / U_SCALE, -32767, 32767)).astype(np.int16))
        signals.append(np.asarray(state.world.concentration[..., 1], np.float32).T)

    snapshot()
    snapshot_fields()
    for step in range(1, steps + 1):
        state, counter = step_sim(state, cfg, counter)
        snapshot()
        if step % field_every == 0:
            snapshot_fields()

    signal = np.maximum(np.stack(signals), 0.0)
    positive = signal[signal > 1e-3]
    signal_max = float(np.percentile(positive, 99.5)) if positive.size else 1.0
    # Square-root encoding keeps resolution in the faint diffusion tails.
    signal_u8 = np.round(255 * np.sqrt(np.clip(signal / signal_max, 0, 1))).astype(np.uint8)

    velocity = np.stack(fields)
    ux, uy = (velocity[..., 0] * U_SCALE), (velocity[..., 1] * U_SCALE)
    solid = state.lbm.mask.astype(np.uint8)  # already [y, x]
    fluid = (solid == 0)
    interior = fluid & np.roll(fluid, 2, 0) & np.roll(fluid, -2, 0) \
        & np.roll(fluid, 2, 1) & np.roll(fluid, -2, 1)
    vorticity = (np.gradient(uy, axis=2) - np.gradient(ux, axis=1))[:, interior]
    meta = {
        "scene": scene, "seed": cfg.seed, "grid": [cfg.world.width, cfg.world.height],
        "steps": steps, "field_every": field_every, "u_scale": U_SCALE,
        "lbm_substeps": cfg.world.lbm_substeps, "viscosity": cfg.world.lbm_viscosity,
        "agent_drag": cfg.world.lbm_agent_drag, "coupling": cfg.world.lbm_agent_coupling,
        "dipole_length": cfg.world.lbm_dipole_length,
        "flow_movement_scale": cfg.embodied.flow_movement_scale,
        "signal_max": signal_max, "population_slots": cfg.embodied.population_size,
        "speed_max": float(np.percentile(np.hypot(ux, uy)[:, fluid], 99.5)),
        "vorticity_max": float(np.percentile(np.abs(vorticity), 99.0)),
        "controllers": "random initial genomes, mutated at reproduction; untrained",
    }
    arrays = {
        "velocity": (velocity, "delta-shuffle"),
        "signal": (signal_u8, "raw"),
        "solid": (solid, "raw"),
        **{key: (np.stack(value), "raw") for key, value in rows.items()},
    }
    return meta, arrays


def export(path: str | Path, scene: str = "obstacle", **kwargs) -> Path:
    config_keys = ("grid", "population", "seed", "substeps", "drag")
    cfg = viewer_config(scene, **{k: kwargs.pop(k) for k in config_keys if k in kwargs})
    return write_replay(path, *record(cfg, scene, **kwargs))
