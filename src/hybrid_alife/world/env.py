"""Shared 2D world model with proxy microfluidic-inspired sensory fields.

All fields are pure JAX arrays. The proxy physics is intentionally cheap and
stylized: no CFD, no wet lab. The goal is to generate a structured,
partially-observable sensory ecology that creates evolutionary pressure for
the six sixth-sense modalities. See `docs/world_model.md` for the full design.

Fields exposed on `WorldState`:

  terrain        [H, W, 4]   structural bias / ridge / basin / channel-mask
  resources      [H, W, R]
  hazards        [H, W, Z]
  flow           [H, W, 2]   (u, v) per-cell flow vector
  curvature      [H, W, 1]   signed channel curvature kappa
  shear          [H, W, 2]   off-diagonal shear-tensor components (du/dy, dv/dx)
  shear_grad     [H, W, 2]   spatial gradient of |shear|
  enrichment     [H, W, 1]   margination/enrichment proxy
  lift           [H, W, 2]   inertial-lift proxy with wall repulsion
  concentration  [H, W, C]   diffusing + advected scalar wave (deposits)
  metabolites    [H, W, M]   diffusing slower scalar (excretions)
  occupancy      [H, W]      living agent id per cell (-1 empty)
  time           scalar i32
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
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    terrain = _build_terrain(h, w)
    resources = jax.random.uniform(k1, (h, w, cfg.resource_channels), minval=0.0, maxval=1.0)
    hazards = 0.05 * jax.random.uniform(k2, (h, w, cfg.hazard_channels), minval=0.0, maxval=1.0)

    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij"
    )
    curvature = build_curvature_map(cfg, xx, yy, k5)
    flow = _build_primary_flow(curvature, xx, yy, cfg)
    noise = cfg.flow_noise_std * jax.random.normal(k3, flow.shape, dtype=jnp.float32)
    flow = flow + noise
    shear = _compute_shear(flow)
    shear_grad = _compute_shear_gradient(shear)
    lift = _compute_lift_proxy(flow, shear, curvature, cfg)
    enrichment = _compute_enrichment_proxy(resources, hazards, lift, None, cfg)
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
# Curvature regimes
# ---------------------------------------------------------------------------


def build_curvature_map(
    cfg: WorldConfig, xx: jax.Array, yy: jax.Array, key: PRNGKey
) -> jax.Array:
    """Build a [H, W, 1] curvature map for the configured regime.

    Each regime is a stylized analogue of a microfluidic channel layout:

      - "default"        legacy sin(2 pi x) * cos(pi y) pattern
      - "straight"       near-zero curvature
      - "spiral"         radial spiral with k full turns
      - "serpentine"     sin wave along x with wavenumber k
      - "obstacle"       circular obstacle producing a wake-like curvature
      - "random_smooth"  low-pass random field
    """
    regime = (cfg.curvature_regime or "default").lower()
    amp = float(cfg.curvature_amplitude)
    k = float(cfg.curvature_wavenumber)
    if regime == "default":
        curv = jnp.sin(2.0 * jnp.pi * xx) * jnp.cos(jnp.pi * yy)
    elif regime == "straight":
        curv = 0.05 * jnp.sin(0.5 * jnp.pi * xx)
    elif regime == "spiral":
        r = jnp.sqrt(xx**2 + yy**2)
        theta = jnp.arctan2(yy, xx)
        curv = jnp.sin(k * theta + 4.0 * r)
    elif regime == "serpentine":
        curv = jnp.sin(k * jnp.pi * xx) * (1.0 - 0.5 * yy**2)
    elif regime == "obstacle":
        r = jnp.sqrt(xx**2 + yy**2)
        bell = jnp.exp(-(r**2) * 6.0)
        wake = jnp.sin(3.0 * jnp.pi * xx) * jnp.exp(-(yy**2) * 4.0)
        curv = bell + 0.4 * wake
    elif regime == "random_smooth":
        raw = jax.random.normal(key, xx.shape, dtype=jnp.float32)
        for _ in range(4):
            raw = 0.2 * (
                raw
                + jnp.roll(raw, 1, axis=0)
                + jnp.roll(raw, -1, axis=0)
                + jnp.roll(raw, 1, axis=1)
                + jnp.roll(raw, -1, axis=1)
            )
        mx = jnp.maximum(jnp.max(jnp.abs(raw)), 1e-6)
        curv = raw / mx
    else:
        raise ValueError(f"unknown curvature_regime={regime!r}")
    return (amp * curv).astype(jnp.float32)[..., None]


# ---------------------------------------------------------------------------
# Field operators
# ---------------------------------------------------------------------------


def _build_primary_flow(
    curvature: jax.Array, xx: jax.Array, yy: jax.Array, cfg: WorldConfig
) -> jax.Array:
    """Combine the regime-dependent primary flow with a Dean-like secondary."""
    primary = _compute_dean_like_flow(curvature, xx, yy)
    secondary = dean_secondary_flow(curvature) * cfg.dean_secondary_strength
    return primary + secondary


def _compute_dean_like_flow(curvature: jax.Array, xx: jax.Array, yy: jax.Array) -> jax.Array:
    """Curvature-induced primary flow.

    Constructs a rotation-like field whose curl is proportional to local
    curvature, so high-curvature regions get strong rotational shear.
    """
    cx = -curvature[..., 0] * yy
    cy = curvature[..., 0] * xx
    return jnp.stack([cx, cy], axis=-1).astype(jnp.float32)


def dean_secondary_flow(curvature: jax.Array) -> jax.Array:
    """Stylized Dean-like secondary (cross-section) flow.

    Real Dean vortices are 3D pairs in the channel cross-section; in 2D we
    proxy them with a divergence-free flow built from the skew gradient of
    curvature: (du, dv) = (-d kappa/dy, d kappa/dx). Its strength scales
    naturally with curvature gradient (Dean number ~ kappa).
    """
    kappa = curvature[..., 0]
    grad_y = 0.5 * (jnp.roll(kappa, -1, axis=0) - jnp.roll(kappa, 1, axis=0))
    grad_x = 0.5 * (jnp.roll(kappa, -1, axis=1) - jnp.roll(kappa, 1, axis=1))
    return jnp.stack([-grad_y, grad_x], axis=-1).astype(jnp.float32)


def _compute_shear(flow: jax.Array) -> jax.Array:
    """Off-diagonal shear-tensor components (du/dy, dv/dx).

    For a 2D incompressible flow, these dominate the symmetric strain-rate
    tensor. We keep the [H, W, 2] layout for backwards compatibility.
    """
    dfx_dy = 0.5 * (jnp.roll(flow[..., 0], -1, axis=0) - jnp.roll(flow[..., 0], 1, axis=0))
    dfy_dx = 0.5 * (jnp.roll(flow[..., 1], -1, axis=1) - jnp.roll(flow[..., 1], 1, axis=1))
    return jnp.stack([dfx_dy, dfy_dx], axis=-1)


def shear_magnitude(shear: jax.Array) -> jax.Array:
    """Scalar shear magnitude sqrt(s_xy^2 + s_yx^2). Returns [H, W]."""
    return jnp.linalg.norm(shear, axis=-1)


def _compute_shear_gradient(shear: jax.Array) -> jax.Array:
    mag = shear_magnitude(shear)
    grad_y = 0.5 * (jnp.roll(mag, -1, axis=0) - jnp.roll(mag, 1, axis=0))
    grad_x = 0.5 * (jnp.roll(mag, -1, axis=1) - jnp.roll(mag, 1, axis=1))
    return jnp.stack([grad_y, grad_x], axis=-1)


def _wall_repulsion(h: int, w: int, strength: float) -> jax.Array:
    """[H, W, 2] field pointing away from non-toroidal walls.

    Tent-shaped falloff: zero in the interior, growing toward walls.
    """
    yy = jnp.linspace(-1.0, 1.0, h)
    xx = jnp.linspace(-1.0, 1.0, w)
    dist_y = 1.0 - jnp.abs(yy)
    dist_x = 1.0 - jnp.abs(xx)
    rep_y = -jnp.sign(yy)
    rep_x = -jnp.sign(xx)
    falloff_y = jnp.clip(1.0 - 4.0 * dist_y, 0.0, 1.0)
    falloff_x = jnp.clip(1.0 - 4.0 * dist_x, 0.0, 1.0)
    field_y = (rep_y * falloff_y)[:, None] * jnp.ones((1, w))
    field_x = (rep_x * falloff_x)[None, :] * jnp.ones((h, 1))
    return strength * jnp.stack([field_y, field_x], axis=-1).astype(jnp.float32)


def _compute_lift_proxy(
    flow: jax.Array, shear: jax.Array, curvature: jax.Array, cfg: WorldConfig
) -> jax.Array:
    """Inertial-lift proxy with optional wall repulsion.

    The lateral lift on a finite-size particle in inertial microfluidics is
    a balance of a shear-gradient term and a wall-induced term. We approximate:

       lift  ~  skew(shear) * U * (1 + |kappa|) * (eps + |shear|)
                + wall_repulsion (if non-toroidal)
    """
    speed = jnp.linalg.norm(flow, axis=-1, keepdims=True)
    shear_mag = jnp.linalg.norm(shear, axis=-1, keepdims=True)
    direction = jnp.concatenate([shear[..., 1:2], -shear[..., 0:1]], axis=-1)
    base = direction * speed * (1.0 + jnp.abs(curvature)) * (0.1 + shear_mag)
    if (not cfg.toroidal) and cfg.wall_repulsion_strength > 0.0:
        h, w = flow.shape[0], flow.shape[1]
        base = base + _wall_repulsion(h, w, cfg.wall_repulsion_strength)
    return base


def _compute_enrichment_proxy(
    resources: jax.Array,
    hazards: jax.Array,
    lift: jax.Array,
    occupancy: jax.Array | None,
    cfg: WorldConfig,
) -> jax.Array:
    """Margination/enrichment scalar.

    Higher when resources dominate hazards, boosted by local inertial-lift
    magnitude, optionally penalized by crowding pressure from occupancy.
    """
    resource_mass = jnp.sum(resources, axis=-1, keepdims=True)
    hazard_mass = jnp.sum(hazards, axis=-1, keepdims=True)
    lift_mag = jnp.linalg.norm(lift, axis=-1, keepdims=True)
    raw = resource_mass - hazard_mass + 0.25 * lift_mag
    if cfg.enrichment_uses_occupancy and occupancy is not None:
        occ_mass = (occupancy >= 0).astype(jnp.float32)[..., None]
        raw = raw - 0.15 * occ_mass
    return jnp.tanh(raw)


# ---------------------------------------------------------------------------
# Advection
# ---------------------------------------------------------------------------


def advect_field(
    field: jax.Array, flow: jax.Array, dt: float, toroidal: bool
) -> jax.Array:
    """Semi-Lagrangian bilinear advection of `field` by `flow`.

    `field` is [H, W, C], `flow` is [H, W, 2]. Returns the same shape.
    Velocity units are grid-cells-per-unit-time scaled by `dt`.
    """
    h, w = field.shape[0], field.shape[1]
    yy, xx = jnp.meshgrid(
        jnp.arange(h, dtype=jnp.float32),
        jnp.arange(w, dtype=jnp.float32),
        indexing="ij",
    )
    sample_y = yy - dt * flow[..., 0]
    sample_x = xx - dt * flow[..., 1]
    if toroidal:
        sample_y = jnp.mod(sample_y, h)
        sample_x = jnp.mod(sample_x, w)
    else:
        sample_y = jnp.clip(sample_y, 0.0, h - 1.0)
        sample_x = jnp.clip(sample_x, 0.0, w - 1.0)

    y0 = jnp.floor(sample_y).astype(jnp.int32)
    x0 = jnp.floor(sample_x).astype(jnp.int32)
    if toroidal:
        y1 = jnp.mod(y0 + 1, h)
        x1 = jnp.mod(x0 + 1, w)
    else:
        y1 = jnp.minimum(y0 + 1, h - 1)
        x1 = jnp.minimum(x0 + 1, w - 1)
    fy = (sample_y - y0.astype(jnp.float32))[..., None]
    fx = (sample_x - x0.astype(jnp.float32))[..., None]

    f00 = field[y0, x0]
    f01 = field[y0, x1]
    f10 = field[y1, x0]
    f11 = field[y1, x1]
    top = f00 * (1.0 - fx) + f01 * fx
    bot = f10 * (1.0 - fx) + f11 * fx
    return top * (1.0 - fy) + bot * fy


# ---------------------------------------------------------------------------
# World stepping
# ---------------------------------------------------------------------------


def step_world(state: SimState, cfg: WorldConfig) -> SimState:
    """Advance passive world fields (flow drift, advection, decay, regen)."""
    world = state.world
    state.rng, k_drift = jax.random.split(state.rng)

    if cfg.drifting:
        flow = _drift_flow(world.flow, world.time, cfg, k_drift)
        shear = _compute_shear(flow)
        shear_grad = _compute_shear_gradient(shear)
        lift = _compute_lift_proxy(flow, shear, world.curvature, cfg)
    else:
        flow = world.flow
        shear = world.shear
        shear_grad = world.shear_grad
        lift = world.lift

    concentration = diffuse_decay(
        world.concentration,
        decay=cfg.concentration_decay,
        diffusion=cfg.concentration_diffusion,
        toroidal=cfg.toroidal,
    )
    if cfg.flow_advection_strength > 0.0:
        concentration = advect_field(
            concentration, flow, cfg.flow_advection_strength, cfg.toroidal
        )
    metabolites = diffuse_decay(
        world.metabolites,
        decay=cfg.metabolite_decay,
        diffusion=cfg.metabolite_diffusion,
        toroidal=cfg.toroidal,
    )

    resources = jnp.clip(world.resources + cfg.resource_regen, 0.0, 1.0)
    hazards = world.hazards * cfg.hazard_decay

    enrichment = _compute_enrichment_proxy(resources, hazards, lift, world.occupancy, cfg)

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


def _drift_flow(
    flow: jax.Array, time: jax.Array, cfg: WorldConfig, key: PRNGKey
) -> jax.Array:
    """Smooth temporal drift: rotation + sinusoidal modulation + Gaussian jitter."""
    t = time.astype(jnp.float32)
    angle = cfg.drift_rotation_rate * t
    c = jnp.cos(angle)
    s = jnp.sin(angle)
    u = flow[..., 0]
    v = flow[..., 1]
    rotated = jnp.stack([c * u - s * v, s * u + c * v], axis=-1)
    mod = 1.0 + 0.1 * jnp.sin(cfg.drift_modulation_freq * t)
    perturb = cfg.drift_speed * jax.random.normal(key, flow.shape, dtype=jnp.float32)
    return 0.99 * rotated * mod + perturb


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


def compute_density(
    positions: jax.Array, alive: jax.Array, h: int, w: int
) -> jax.Array:
    """Per-cell living-agent count, [H, W] float32.

    Unlike `compute_occupancy`, this returns the additive count of agents
    sharing a cell and so represents real local hemodynamic crowding.
    """
    grid = positions_to_grid(positions, h, w)
    flat_idx = grid[..., 0] * w + grid[..., 1]
    counts = jnp.zeros((h * w,), dtype=jnp.float32)
    counts = counts.at[flat_idx].add(alive.astype(jnp.float32))
    return counts.reshape((h, w))


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
