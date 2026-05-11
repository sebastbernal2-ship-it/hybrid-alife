"""Shared 2D world model and environment transition skeleton."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from hybrid_alife.types import PRNGKey, SimState, WorldConfig, WorldState


def initialize_world(cfg: WorldConfig, key: PRNGKey) -> WorldState:
    """Create initial proxy-physics fields and ecological channels."""
    h, w = cfg.height, cfg.width
    k1, k2, k3, k4 = jax.random.split(key, 4)

    terrain = jnp.zeros((h, w, 4), dtype=jnp.float32)
    resources = jax.random.uniform(k1, (h, w, cfg.resource_channels), minval=0.0, maxval=1.0)
    hazards = 0.05 * jax.random.uniform(k2, (h, w, cfg.hazard_channels), minval=0.0, maxval=1.0)

    yy, xx = jnp.meshgrid(jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij")
    curvature = jnp.sin(2.0 * jnp.pi * xx)[..., None] * jnp.cos(jnp.pi * yy)[..., None]
    flow = _compute_dean_like_flow(curvature, xx, yy)
    shear = _compute_shear(flow)
    lift = _compute_lift_proxy(flow, shear, curvature)
    enrichment = _compute_enrichment_proxy(resources, hazards, lift)
    concentration = jax.random.normal(k3, (h, w, 2), dtype=jnp.float32) * 0.01
    occupancy = -jnp.ones((h, w), dtype=jnp.int32)

    noise = cfg.flow_noise_std * jax.random.normal(k4, flow.shape, dtype=jnp.float32)
    flow = flow + noise

    return WorldState(
        terrain=terrain,
        resources=resources,
        hazards=hazards,
        flow=flow,
        curvature=curvature.astype(jnp.float32),
        shear=shear.astype(jnp.float32),
        enrichment=enrichment.astype(jnp.float32),
        lift=lift.astype(jnp.float32),
        concentration=concentration,
        occupancy=occupancy,
    )


def step_world(state: SimState, cfg: WorldConfig) -> SimState:
    """Advance passive world fields.

    V1 updates concentration waves only. Later versions should couple agent actions,
    resource regeneration, hazard diffusion, and branch-specific deposits.
    """
    world = state.world
    concentration = diffuse_decay(
        world.concentration,
        decay=cfg.concentration_decay,
        diffusion=cfg.concentration_diffusion,
        toroidal=cfg.toroidal,
    )
    state.world.concentration = concentration
    return state


def diffuse_decay(x: jax.Array, decay: float, diffusion: float, toroidal: bool) -> jax.Array:
    """Simple five-point stencil diffusion plus decay."""
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


def _compute_dean_like_flow(curvature: jax.Array, xx: jax.Array, yy: jax.Array) -> jax.Array:
    """Proxy for curvature-induced secondary flow, not a CFD solution."""
    cx = -curvature[..., 0] * yy
    cy = curvature[..., 0] * xx
    return jnp.stack([cx, cy], axis=-1).astype(jnp.float32)


def _compute_shear(flow: jax.Array) -> jax.Array:
    dfx_dy = 0.5 * (jnp.roll(flow[..., 0], -1, axis=0) - jnp.roll(flow[..., 0], 1, axis=0))
    dfy_dx = 0.5 * (jnp.roll(flow[..., 1], -1, axis=1) - jnp.roll(flow[..., 1], 1, axis=1))
    return jnp.stack([dfx_dy, dfy_dx], axis=-1)


def _compute_lift_proxy(flow: jax.Array, shear: jax.Array, curvature: jax.Array) -> jax.Array:
    speed = jnp.linalg.norm(flow, axis=-1, keepdims=True)
    shear_mag = jnp.linalg.norm(shear, axis=-1, keepdims=True)
    direction = jnp.concatenate([shear[..., 1:2], -shear[..., 0:1]], axis=-1)
    return direction * speed * (1.0 + jnp.abs(curvature)) * (0.1 + shear_mag)


def _compute_enrichment_proxy(resources: jax.Array, hazards: jax.Array, lift: jax.Array) -> jax.Array:
    resource_mass = jnp.sum(resources, axis=-1, keepdims=True)
    hazard_mass = jnp.sum(hazards, axis=-1, keepdims=True)
    lift_mag = jnp.linalg.norm(lift, axis=-1, keepdims=True)
    return jnp.tanh(resource_mass - hazard_mass + 0.25 * lift_mag)

