"""Six sixth-sense modalities, sampled at each agent's location.

Each sense returns a [N, k] tensor. The full sixth-sense observation is
the concatenation of all six modalities. The modalities are:

  1. Dean-flow curvature scalar         : [N, 1]
  2. Shear gradient vector              : [N, 2]
  3. Margination / enrichment scalar    : [N, 1]
  4. Inertial lift vector               : [N, 2]
  5. Concentration-wave channels        : [N, C]
  6. Hemodynamic / crowding context     : [N, 3]  (local crowd density, mean flow speed, neighbor msg energy)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import EmbodiedConfig, PRNGKey, WorldConfig, WorldState
from hybrid_alife.world.env import positions_to_grid


SIXTH_SENSE_LABELS = (
    "curvature",
    "shear_grad",
    "enrichment",
    "lift",
    "concentration",
    "crowd_ctx",
)


def sixth_sense_dim(world_cfg: WorldConfig) -> int:
    """1 + 2 + 1 + 2 + C + 3."""
    return 1 + 2 + 1 + 2 + 2 + 3  # concentration always 2 channels in init


def sample_field_at(field: jax.Array, positions: jax.Array, h: int, w: int) -> jax.Array:
    """Sample a [H, W, C] field at each agent's cell -> [N, C]."""
    grid = positions_to_grid(positions, h, w)
    return field[grid[:, 0], grid[:, 1]]


def crowding_context(
    positions: jax.Array,
    alive: jax.Array,
    messages: jax.Array,
    flow: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Compute simple crowding context for each agent.

    Returns [N, 3]: (neighbor_count_proxy, mean_flow_speed_at_cell, neighbor_msg_energy).
    """
    grid = positions_to_grid(positions, h, w)
    # neighbor_count_proxy = number of agents in same cell (including self) minus 1.
    flat = grid[:, 0] * w + grid[:, 1]
    counts = jnp.zeros((h * w,), dtype=jnp.float32)
    counts = counts.at[flat].add(alive.astype(jnp.float32))
    per_agent_count = jnp.maximum(counts[flat] - 1.0, 0.0)

    speed = jnp.linalg.norm(flow[grid[:, 0], grid[:, 1]], axis=-1)

    msg_energy = jnp.sum(messages * messages, axis=-1)
    msg_per_cell = jnp.zeros((h * w,), dtype=jnp.float32)
    msg_per_cell = msg_per_cell.at[flat].add(msg_energy * alive.astype(jnp.float32))
    neighbor_msg = jnp.maximum(msg_per_cell[flat] - msg_energy * alive.astype(jnp.float32), 0.0)

    return jnp.stack([per_agent_count, speed, neighbor_msg], axis=-1)


def sample_sixth_sense(
    world: WorldState,
    positions: jax.Array,
    alive: jax.Array,
    messages: jax.Array,
    world_cfg: WorldConfig,
    embodied_cfg: EmbodiedConfig,
    key: PRNGKey,
) -> jax.Array:
    """Return [N, sixth_sense_dim] tensor of sensory readings.

    Ablation knobs:
      - blind: return zeros.
      - true_sixth_sense=False: drop all sixth-sense channels (zeros).
      - shuffle_sixth_sense: per-step permutation of the sixth-sense slice (destroys semantics).
      - sensory_noise_std: additive Gaussian noise.
    """
    h, w = world_cfg.height, world_cfg.width
    n = positions.shape[0]

    if embodied_cfg.blind:
        return jnp.zeros((n, sixth_sense_dim(world_cfg)), dtype=jnp.float32)

    curv = sample_field_at(world.curvature, positions, h, w)  # [N, 1]
    sgrad = sample_field_at(world.shear_grad, positions, h, w)  # [N, 2]
    enr = sample_field_at(world.enrichment, positions, h, w)  # [N, 1]
    lft = sample_field_at(world.lift, positions, h, w)  # [N, 2]
    conc = sample_field_at(world.concentration, positions, h, w)  # [N, 2]
    crowd = crowding_context(positions, alive, messages, world.flow, h, w)  # [N, 3]

    parts = [curv, sgrad, enr, lft, conc, crowd]
    if not embodied_cfg.true_sixth_sense:
        parts = [jnp.zeros_like(p) for p in parts]

    sense = jnp.concatenate(parts, axis=-1)

    if embodied_cfg.shuffle_sixth_sense:
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, sense.shape[-1])
        sense = sense[:, perm]

    key, k_noise = jax.random.split(key)
    sense = sense + world_cfg.sensory_noise_std * jax.random.normal(
        k_noise, sense.shape, dtype=jnp.float32
    )
    return sense.astype(jnp.float32)
