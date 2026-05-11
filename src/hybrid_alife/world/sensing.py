"""Six sixth-sense modalities, sampled at each agent's location.

Each modality is returned as a [N, k] tensor. The full sixth-sense observation
is the concatenation of all six. The layout is fixed (callers depend on it):

  1. Dean-flow curvature scalar                       [N, 1]
  2. Shear gradient vector                            [N, 2]
  3. Margination / enrichment scalar                  [N, 1]
  4. Inertial lift vector                             [N, 2]
  5. Concentration-wave channels                      [N, 2]
  6. Hemodynamic / crowding context                   [N, 3]
       (local density, mean flow speed, neighbor msg energy)

Ablation / uncertainty modes (controlled by EmbodiedConfig):

  - blind                : return all zeros
  - true_sixth_sense=F   : zero out all sixth-sense channels
  - shuffle_sixth_sense  : per-step random permutation across channels
  - sense_delayed        : sample upstream-of-flow (stale reading proxy)
  - sense_multiplicative_noise_std > 0 : heteroskedastic multiplicative noise
  - sense_dropout_frac > 0             : random channel dropout
  - WorldConfig.sensory_noise_std      : additive Gaussian noise
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import EmbodiedConfig, PRNGKey, WorldConfig, WorldState
from hybrid_alife.world.env import compute_density, positions_to_grid

SIXTH_SENSE_LABELS = (
    "curvature",
    "shear_grad",
    "enrichment",
    "lift",
    "concentration",
    "crowd_ctx",
)
# Per-modality channel widths used to slice the concatenated tensor.
SIXTH_SENSE_WIDTHS = (1, 2, 1, 2, 2, 3)


def sixth_sense_dim(world_cfg: WorldConfig) -> int:
    """1 + 2 + 1 + 2 + 2 + 3 = 11."""
    return sum(SIXTH_SENSE_WIDTHS)


def sample_field_at(field: jax.Array, positions: jax.Array, h: int, w: int) -> jax.Array:
    """Sample a [H, W, C] field at each agent's cell -> [N, C]."""
    grid = positions_to_grid(positions, h, w)
    return field[grid[:, 0], grid[:, 1]]


def _sample_field_delayed(
    field: jax.Array,
    positions: jax.Array,
    flow: jax.Array,
    delay_steps: int,
    h: int,
    w: int,
) -> jax.Array:
    """Sample a field at a position trace upstream by `delay_steps` of flow.

    This is a cheap proxy for stale sensory readings: the agent perceives the
    field at the upstream cell the fluid arrived from, so the higher the
    delay, the more advected (and stale) the reading.
    """
    grid = positions_to_grid(positions, h, w)
    cell_flow = flow[grid[:, 0], grid[:, 1]]  # [N, 2]
    # walk upstream by delay_steps; toroidal modulus keeps indices in range
    fy = (grid[:, 0].astype(jnp.float32) - delay_steps * cell_flow[:, 0]) % h
    fx = (grid[:, 1].astype(jnp.float32) - delay_steps * cell_flow[:, 1]) % w
    ry = jnp.clip(fy.astype(jnp.int32), 0, h - 1)
    rx = jnp.clip(fx.astype(jnp.int32), 0, w - 1)
    return field[ry, rx]


def crowding_context(
    positions: jax.Array,
    alive: jax.Array,
    messages: jax.Array,
    flow: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Compute the local hemodynamic / crowding context for each agent.

    Returns [N, 3]:
      0. local density (other agents at same cell)
      1. mean flow speed at the agent's cell (laminar / turbulent context)
      2. neighbor message energy at the cell (excluding self)

    Density and message-energy are computed with O(N) scatter-adds, so the
    operator is vectorized across N.
    """
    grid = positions_to_grid(positions, h, w)
    flat = grid[:, 0] * w + grid[:, 1]
    counts = jnp.zeros((h * w,), dtype=jnp.float32)
    counts = counts.at[flat].add(alive.astype(jnp.float32))
    per_agent_count = jnp.maximum(counts[flat] - 1.0, 0.0)

    speed = jnp.linalg.norm(flow[grid[:, 0], grid[:, 1]], axis=-1)

    msg_energy = jnp.sum(messages * messages, axis=-1)
    msg_per_cell = jnp.zeros((h * w,), dtype=jnp.float32)
    msg_per_cell = msg_per_cell.at[flat].add(msg_energy * alive.astype(jnp.float32))
    neighbor_msg = jnp.maximum(
        msg_per_cell[flat] - msg_energy * alive.astype(jnp.float32), 0.0
    )

    return jnp.stack([per_agent_count, speed, neighbor_msg], axis=-1)


def occupancy_pressure(
    positions: jax.Array,
    alive: jax.Array,
    flow: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Scalar occupancy pressure per agent: density * mean flow speed.

    This proxies the "crowding under flow" pressure that drives margination in
    hemodynamic systems: high density on top of high local shear/flow speed
    amplifies effective pressure.
    """
    density = compute_density(positions, alive, h, w)
    speed_grid = jnp.linalg.norm(flow, axis=-1)
    grid = positions_to_grid(positions, h, w)
    return density[grid[:, 0], grid[:, 1]] * speed_grid[grid[:, 0], grid[:, 1]]


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
      - blind                       : return zeros
      - true_sixth_sense=False      : drop all sixth-sense channels (zeros)
      - shuffle_sixth_sense         : per-step permutation across channels
      - sense_delayed               : sample upstream (stale reading proxy)
      - sense_multiplicative_noise_std > 0 : heteroskedastic noise
      - sense_dropout_frac > 0      : random channel dropout
      - WorldConfig.sensory_noise_std : additive Gaussian noise
    """
    h, w = world_cfg.height, world_cfg.width
    n = positions.shape[0]

    if embodied_cfg.blind:
        return jnp.zeros((n, sixth_sense_dim(world_cfg)), dtype=jnp.float32)

    if embodied_cfg.sense_delayed and embodied_cfg.sense_delay_steps > 0:
        d = int(embodied_cfg.sense_delay_steps)
        flow = world.flow
        curv = _sample_field_delayed(world.curvature, positions, flow, d, h, w)
        sgrad = _sample_field_delayed(world.shear_grad, positions, flow, d, h, w)
        enr = _sample_field_delayed(world.enrichment, positions, flow, d, h, w)
        lft = _sample_field_delayed(world.lift, positions, flow, d, h, w)
        conc = _sample_field_delayed(world.concentration, positions, flow, d, h, w)
    else:
        curv = sample_field_at(world.curvature, positions, h, w)
        sgrad = sample_field_at(world.shear_grad, positions, h, w)
        enr = sample_field_at(world.enrichment, positions, h, w)
        lft = sample_field_at(world.lift, positions, h, w)
        conc = sample_field_at(world.concentration, positions, h, w)
    crowd = crowding_context(positions, alive, messages, world.flow, h, w)

    parts = [curv, sgrad, enr, lft, conc, crowd]
    if not embodied_cfg.true_sixth_sense:
        parts = [jnp.zeros_like(p) for p in parts]

    sense = jnp.concatenate(parts, axis=-1)

    if embodied_cfg.shuffle_sixth_sense:
        key, k_perm = jax.random.split(key)
        perm = jax.random.permutation(k_perm, sense.shape[-1])
        sense = sense[:, perm]

    if embodied_cfg.sense_multiplicative_noise_std > 0.0:
        key, k_mult = jax.random.split(key)
        mult = 1.0 + embodied_cfg.sense_multiplicative_noise_std * jax.random.normal(
            k_mult, sense.shape, dtype=jnp.float32
        )
        sense = sense * mult

    if embodied_cfg.sense_dropout_frac > 0.0:
        key, k_drop = jax.random.split(key)
        keep_prob = 1.0 - float(embodied_cfg.sense_dropout_frac)
        mask = (jax.random.uniform(k_drop, (sense.shape[-1],)) < keep_prob).astype(
            jnp.float32
        )
        sense = sense * mask[None, :]

    key, k_noise = jax.random.split(key)
    sense = sense + world_cfg.sensory_noise_std * jax.random.normal(
        k_noise, sense.shape, dtype=jnp.float32
    )
    return sense.astype(jnp.float32)
