"""Opt-in egocentric senses (EmbodiedConfig.compact_senses).

Ten channels in the same (row, column) frame as the movement outputs, so that a
controller can evolve to steer by them. They are raw senses, not behaviour: nothing
here moves a body, and founder weights are random.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import EmbodiedConfig, EmbodiedPopulationState, WorldConfig, WorldState

COMPACT_SENSE_LABELS = (
    "food_row",
    "food_col",
    "food_here",
    "other_row",
    "other_col",
    "other_size",
    "hazard_row",
    "hazard_col",
    "hazard_here",
    "energy",
)
COMPACT_SENSE_DIM = len(COMPACT_SENSE_LABELS)
# A cell counts as food (or hazard) for direction sensing above this level.
_PRESENT = 0.15


def _nearest_cell(
    field: jax.Array, positions: jax.Array, radius: int, toroidal: bool
) -> tuple[jax.Array, jax.Array]:
    """Direction to the nearest present cell, scaled by closeness; and the level underfoot.

    ``field`` is [H, W]. Returns ([N, 2], [N]).
    """
    h, w = field.shape
    scaled = positions * jnp.array([h, w])
    cell = jnp.clip(jnp.floor(scaled).astype(jnp.int32), 0, jnp.array([h - 1, w - 1]))
    offsets = jnp.arange(-radius, radius + 1)
    oy, ox = jnp.meshgrid(offsets, offsets, indexing="ij")
    ry = cell[:, 0, None] + oy.reshape(-1)[None]
    rx = cell[:, 1, None] + ox.reshape(-1)[None]
    inside = jnp.ones_like(ry, dtype=bool)
    if not toroidal:
        inside = (ry >= 0) & (ry < h) & (rx >= 0) & (rx < w)
    level = field[ry % h, rx % w]
    delta = jnp.stack(
        [ry + 0.5 - scaled[:, 0, None], rx + 0.5 - scaled[:, 1, None]], axis=-1
    )
    dist = jnp.linalg.norm(delta, axis=-1)
    dist = jnp.where((level > _PRESENT) & inside & (dist <= radius), dist, jnp.inf)
    best = jnp.argmin(dist, axis=1)
    rows = jnp.arange(positions.shape[0])
    best_dist = dist[rows, best]
    found = jnp.isfinite(best_dist)
    safe = jnp.where(found, best_dist, 1.0)
    closeness = jnp.where(found, 1.0 - 0.7 * safe / radius, 0.0)
    direction = delta[rows, best] / jnp.maximum(safe, 0.5)[:, None]
    return direction * closeness[:, None], field[cell[:, 0], cell[:, 1]]


def compact_observation(
    pop: EmbodiedPopulationState,
    world: WorldState,
    cfg: EmbodiedConfig,
    world_cfg: WorldConfig,
) -> jax.Array:
    """[N, COMPACT_SENSE_DIM]. ``blind`` zeroes every external channel."""
    n = pop.positions.shape[0]
    radius = cfg.compact_sense_radius
    food_dir, food_here = _nearest_cell(
        world.resources[..., 0], pop.positions, radius, world_cfg.toroidal
    )
    hazard_dir, hazard_here = _nearest_cell(
        world.hazards[..., 0], pop.positions, radius, world_cfg.toroidal
    )

    delta = pop.positions[None, :, :] - pop.positions[:, None, :]  # i -> j
    if world_cfg.toroidal:
        delta = delta - jnp.round(delta)
    cells = delta * jnp.array([world_cfg.height, world_cfg.width])
    dist = jnp.linalg.norm(cells, axis=-1)
    other = pop.alive[None, :] & ~jnp.eye(n, dtype=bool)
    dist = jnp.where(other & (dist <= radius), dist, jnp.inf)
    nearest = jnp.argmin(dist, axis=1)
    rows = jnp.arange(n)
    near_dist = dist[rows, nearest]
    seen = jnp.isfinite(near_dist)
    safe = jnp.where(seen, near_dist, 1.0)
    closeness = jnp.where(seen, 1.0 - 0.7 * safe / radius, 0.0)
    other_dir = cells[rows, nearest] / jnp.maximum(safe, 0.5)[:, None] * closeness[:, None]
    size = pop.genomes.get("body_radius", jnp.ones((n,)))
    size = jnp.clip(size, 0.6, 1.8)
    other_size = jnp.where(seen, size[nearest] - size, 0.0)

    external = jnp.concatenate(
        [
            food_dir,
            food_here[:, None],
            other_dir,
            other_size[:, None],
            hazard_dir,
            hazard_here[:, None],
        ],
        axis=-1,
    )
    if cfg.blind:
        external = jnp.zeros_like(external)
    scale = cfg.max_energy if cfg.max_energy > 0.0 else 2.0 * cfg.initial_energy
    energy = jnp.clip(pop.energy / scale, 0.0, 2.0)
    return jnp.concatenate([external, energy[:, None]], axis=-1).astype(jnp.float32)
