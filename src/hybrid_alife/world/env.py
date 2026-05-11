"""Shared 2D world model with proxy microfluidic-inspired sensory fields.

All fields are pure JAX arrays. The proxy physics is intentionally cheap and stylized:
no CFD, no wet lab. The goal is to generate a structured, partially-observable sensory
ecology that creates evolutionary pressure for the six sixth-sense modalities:

  1. Dean-flow curvature  -> curvature field
  2. Shear gradient       -> spatial derivative of shear magnitude
  3. Margination /
     enrichment proxy     -> enrichment field, biased by lift + concentration
  4. Inertial lift proxy  -> derived from shear, curvature, and local flow speed
  5. Concentration-wave   -> diffusing scalar field released by agents/metabolism
  6. Local hemodynamic /
     crowding context     -> occupancy + flow speed in local patch
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import PRNGKey, SimState, WorldConfig, WorldState


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def initialize_world(cfg: WorldConfig, key: PRNGKey) -> WorldState:
    """Create initial proxy-physics fields and ecological channels."""
    h, w = cfg.height, cfg.width
    k1, k2, k3, k4, _ = jax.random.split(key, 5)

    terrain = _build_terrain(h, w)
    resources = jax.random.uniform(k1, (h, w, cfg.resource_channels), minval=0.0, maxval=1.0)
    hazards = 0.05 * jax.random.uniform(k2, (h, w, cfg.hazard_channels), minval=0.0, maxval=1.0)

    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij"
    )
    curvature = (jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(jnp.pi * yy))[..., None]
    flow = _compute_dean_like_flow(curvature, xx, yy)
    noise = cfg.flow_noise_std * jax.random.normal(k3, flow.shape, dtype=jnp.float32)
    flow = flow + noise
    shear = _compute_shear(flow)
    shear_grad = _compute_shear_gradient(shear)
    lift = _compute_lift_proxy(flow, shear, curvature)
    enrichment = _compute_enrichment_proxy(resources, hazards, lift)
    concentration = jax.random.normal(k4, (h, w, 2), dtype=jnp.float32) * 0.01
    metabolites = jnp.zeros((h, w, cfg.metabolite_channels), dtype=jnp.float32)
    occupancy = -jnp.ones((h, w), dtype=jnp.int32)

    return WorldState(
        terrain=terrain,
        resources=resources.astype(jnp.float32),
        hazards=hazards.astype(jnp.float32),
        flow=flow.astype(jnp.float32),
        curvature=curvature.astype(jnp.float32),
        shear=shear.astype(jnp.float32),
        shear_grad=shear_grad.astype(jnp.float32),
        enrichment=enrichment.astype(jnp.float32),
        lift=lift.astype(jnp.float32),
        concentration=concentration.astype(jnp.float32),
        metabolites=metabolites,
        occupancy=occupancy,
        time=jnp.asarray(0, dtype=jnp.int32),
    )


def _build_terrain(h: int, w: int) -> jax.Array:
    """4-channel structural terrain: bias, ridge, basin, channel-mask."""
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij"
    )
    ridge = jnp.exp(-(yy**2) * 4.0)
    basin = jnp.exp(-(xx**2 + yy**2) * 2.0)
    channel_mask = (jnp.abs(yy) < 0.6).astype(jnp.float32)
    bias = jnp.ones_like(yy)
    return jnp.stack([bias, ridge, basin, channel_mask], axis=-1).astype(jnp.float32)


# ---------------------------------------------------------------------------
# Field operators
# ---------------------------------------------------------------------------


def _compute_dean_like_flow(curvature: jax.Array, xx: jax.Array, yy: jax.Array) -> jax.Array:
    """Proxy for curvature-induced secondary flow."""
    cx = -curvature[..., 0] * yy
    cy = curvature[..., 0] * xx
    return jnp.stack([cx, cy], axis=-1).astype(jnp.float32)


def _compute_shear(flow: jax.Array) -> jax.Array:
    dfx_dy = 0.5 * (jnp.roll(flow[..., 0], -1, axis=0) - jnp.roll(flow[..., 0], 1, axis=0))
    dfy_dx = 0.5 * (jnp.roll(flow[..., 1], -1, axis=1) - jnp.roll(flow[..., 1], 1, axis=1))
    return jnp.stack([dfx_dy, dfy_dx], axis=-1)


def _compute_shear_gradient(shear: jax.Array) -> jax.Array:
    mag = jnp.linalg.norm(shear, axis=-1)
    grad_y = 0.5 * (jnp.roll(mag, -1, axis=0) - jnp.roll(mag, 1, axis=0))
    grad_x = 0.5 * (jnp.roll(mag, -1, axis=1) - jnp.roll(mag, 1, axis=1))
    return jnp.stack([grad_y, grad_x], axis=-1)


def _compute_lift_proxy(flow: jax.Array, shear: jax.Array, curvature: jax.Array) -> jax.Array:
    speed = jnp.linalg.norm(flow, axis=-1, keepdims=True)
    shear_mag = jnp.linalg.norm(shear, axis=-1, keepdims=True)
    direction = jnp.concatenate([shear[..., 1:2], -shear[..., 0:1]], axis=-1)
    return direction * speed * (1.0 + jnp.abs(curvature)) * (0.1 + shear_mag)


def _compute_enrichment_proxy(
    resources: jax.Array, hazards: jax.Array, lift: jax.Array
) -> jax.Array:
    resource_mass = jnp.sum(resources, axis=-1, keepdims=True)
    hazard_mass = jnp.sum(hazards, axis=-1, keepdims=True)
    lift_mag = jnp.linalg.norm(lift, axis=-1, keepdims=True)
    return jnp.tanh(resource_mass - hazard_mass + 0.25 * lift_mag)


# ---------------------------------------------------------------------------
# World stepping
# ---------------------------------------------------------------------------


def step_world(state: SimState, cfg: WorldConfig) -> SimState:
    """Advance passive world fields."""
    world = state.world
    state.rng, k_drift = jax.random.split(state.rng)

    concentration = diffuse_decay(
        world.concentration,
        decay=cfg.concentration_decay,
        diffusion=cfg.concentration_diffusion,
        toroidal=cfg.toroidal,
    )
    metabolites = diffuse_decay(
        world.metabolites,
        decay=cfg.metabolite_decay,
        diffusion=cfg.metabolite_diffusion,
        toroidal=cfg.toroidal,
    )

    resources = jnp.clip(world.resources + cfg.resource_regen, 0.0, 1.0)
    hazards = world.hazards * cfg.hazard_decay

    if cfg.drifting:
        flow = _drift_flow(world.flow, cfg, k_drift)
        shear = _compute_shear(flow)
        shear_grad = _compute_shear_gradient(shear)
        lift = _compute_lift_proxy(flow, shear, world.curvature)
    else:
        flow = world.flow
        shear = world.shear
        shear_grad = world.shear_grad
        lift = world.lift
    enrichment = _compute_enrichment_proxy(resources, hazards, lift)

    state.world = WorldState(
        terrain=world.terrain,
        resources=resources,
        hazards=hazards,
        flow=flow,
        curvature=world.curvature,
        shear=shear,
        shear_grad=shear_grad,
        enrichment=enrichment,
        lift=lift,
        concentration=concentration,
        metabolites=metabolites,
        occupancy=world.occupancy,
        time=world.time + 1,
    )
    return state


def _drift_flow(flow: jax.Array, cfg: WorldConfig, key: PRNGKey) -> jax.Array:
    """Slow drift / rotation of flow field to simulate non-stationary environment."""
    perturb = cfg.drift_speed * jax.random.normal(key, flow.shape, dtype=jnp.float32)
    return 0.99 * flow + perturb


def diffuse_decay(x: jax.Array, decay: float, diffusion: float, toroidal: bool) -> jax.Array:
    """Five-point stencil diffusion plus decay."""
    if toroidal:
        north = jnp.roll(x, -1, axis=0)
        south = jnp.roll(x, 1, axis=0)
        east = jnp.roll(x, -1, axis=1)
        west = jnp.roll(x, 1, axis=1)
    else:
        north = jnp.concatenate([x[1:], x[-1:]], axis=0)
        south = jnp.concatenate([x[:1], x[:-1]], axis=0)
        east = jnp.concatenate([x[:, 1:], x[:, -1:]], axis=1)
        west = jnp.concatenate([x[:, :1], x[:, :-1]], axis=1)
    lap = north + south + east + west - 4.0 * x
    return decay * (x + diffusion * lap)


# ---------------------------------------------------------------------------
# Position helpers
# ---------------------------------------------------------------------------


def positions_to_grid(positions: jax.Array, h: int, w: int) -> jax.Array:
    """Map continuous [0,1) positions to (row, col) integer indices."""
    ry = jnp.clip(jnp.floor(positions[..., 0] * h).astype(jnp.int32), 0, h - 1)
    rx = jnp.clip(jnp.floor(positions[..., 1] * w).astype(jnp.int32), 0, w - 1)
    return jnp.stack([ry, rx], axis=-1)


def compute_occupancy(
    positions: jax.Array, alive: jax.Array, h: int, w: int
) -> jax.Array:
    """Scatter living agent ids onto a grid; -1 means empty."""
    grid = positions_to_grid(positions, h, w)
    flat_idx = grid[..., 0] * w + grid[..., 1]
    ids = jnp.where(alive, jnp.arange(positions.shape[0], dtype=jnp.int32), -1)
    occ = -jnp.ones((h * w,), dtype=jnp.int32)
    occ = occ.at[flat_idx].set(ids)
    return occ.reshape((h, w))


def gather_patch(
    field: jax.Array, positions: jax.Array, radius: int, h: int, w: int, toroidal: bool
) -> jax.Array:
    """Gather a (2r+1)x(2r+1) patch from a [H, W, C] field for each position.

    Returns: [N, (2r+1)*(2r+1)*C] flat tensor.
    """
    side = 2 * radius + 1
    grid = positions_to_grid(positions, h, w)
    offsets = jnp.arange(-radius, radius + 1)
    oy, ox = jnp.meshgrid(offsets, offsets, indexing="ij")
    ry = grid[:, 0:1, None] + oy[None]
    rx = grid[:, 1:2, None] + ox[None]
    if toroidal:
        ry = jnp.mod(ry, h)
        rx = jnp.mod(rx, w)
    else:
        ry = jnp.clip(ry, 0, h - 1)
        rx = jnp.clip(rx, 0, w - 1)
    patch = field[ry, rx]
    return patch.reshape((positions.shape[0], side * side * field.shape[-1]))


# ---------------------------------------------------------------------------
# Agent-world coupling: deposits and resource consumption
# ---------------------------------------------------------------------------


def scatter_add_to_grid(
    field: jax.Array,
    positions: jax.Array,
    values: jax.Array,
    alive: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Add per-agent values into [H, W, C] field at agent grid cells."""
    grid = positions_to_grid(positions, h, w)
    mask = alive.astype(values.dtype)[:, None]
    masked_values = values * mask
    return field.at[grid[:, 0], grid[:, 1]].add(masked_values)


def consume_from_grid(
    field: jax.Array,
    positions: jax.Array,
    rate: float,
    alive: jax.Array,
    h: int,
    w: int,
) -> tuple[jax.Array, jax.Array]:
    """Each living agent eats a fraction of total resource at its cell.

    Returns the updated field and per-agent consumed scalar [N].
    """
    grid = positions_to_grid(positions, h, w)
    cell_values = field[grid[:, 0], grid[:, 1]]  # [N, C]
    alive_f = alive.astype(field.dtype)
    available = jnp.sum(cell_values, axis=-1)
    take = available * rate * alive_f
    new_cell_values = cell_values * (1.0 - rate * alive_f[:, None])
    field = field.at[grid[:, 0], grid[:, 1]].set(new_cell_values)
    return field, take


def hazard_damage(
    field: jax.Array,
    positions: jax.Array,
    alive: jax.Array,
    h: int,
    w: int,
) -> jax.Array:
    """Sample hazard magnitude at each agent's cell."""
    grid = positions_to_grid(positions, h, w)
    cell_values = field[grid[:, 0], grid[:, 1]]
    return jnp.sum(cell_values, axis=-1) * alive.astype(field.dtype)
