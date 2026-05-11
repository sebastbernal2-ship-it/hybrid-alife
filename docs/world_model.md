# World Model (proxy microfluidic substrate)

This document describes the proxy physics, sensory ecology, and ablation knobs
implemented in `src/hybrid_alife/world/`. All fields are pure JAX arrays and
all kernels are vectorized; the entire pipeline runs on CPU or GPU/TPU with no
external solver.

The goal is *not* to simulate fluid mechanics. It is to produce a partially
observable, structured sensory ecology that creates evolutionary pressure for
the six sixth-sense modalities. Every "physical" quantity below is a
deliberate stylization.

## State containers

`WorldState` is a flat dataclass of arrays:

| Field           | Shape         | Meaning                                              |
| --------------- | ------------- | ---------------------------------------------------- |
| `terrain`       | `[H, W, 4]`   | bias, ridge, basin, channel-mask                     |
| `resources`     | `[H, W, R]`   | regenerating resource channels                       |
| `hazards`       | `[H, W, Z]`   | decaying hazard channels                             |
| `flow`          | `[H, W, 2]`   | (u, v) per-cell flow vector                          |
| `curvature`     | `[H, W, 1]`   | signed channel curvature kappa                       |
| `shear`         | `[H, W, 2]`   | off-diagonal shear tensor (du/dy, dv/dx)             |
| `shear_grad`    | `[H, W, 2]`   | spatial gradient of |shear|                          |
| `enrichment`    | `[H, W, 1]`   | margination/enrichment scalar proxy                  |
| `lift`          | `[H, W, 2]`   | inertial-lift proxy with wall repulsion              |
| `concentration` | `[H, W, C]`   | diffusing + advected scalar wave (agent deposits)    |
| `metabolites`   | `[H, W, M]`   | diffusing slower scalar (excretions)                 |
| `occupancy`     | `[H, W]`      | living agent id per cell, -1 empty                   |
| `time`          | scalar `i32`  |                                                      |

The agent-facing observation is built from a square patch around each agent's
cell plus the six sixth-sense modalities sampled at the cell.

## Curvature regimes

`WorldConfig.curvature_regime` selects a stylized channel layout. Each regime
produces a different `curvature` map, which in turn shapes the primary flow,
the Dean secondary flow, and the inertial-lift proxy.

| regime           | analogue                            | character                                    |
| ---------------- | ----------------------------------- | -------------------------------------------- |
| `default`        | (legacy) sin/cos pattern            | broadband, mixed curvature                   |
| `straight`       | straight channel                    | near-zero curvature, dominated by noise/drift|
| `spiral`         | spiral inertial-focusing channel    | radial spiral with `curvature_wavenumber`    |
| `serpentine`     | serpentine focusing channel         | sin wave along x, decaying off-axis          |
| `obstacle`       | post / pillar in a channel          | bell + downstream wake                       |
| `random_smooth`  | "rough" channel / unstructured      | low-pass random field, seed-controlled       |

`curvature_amplitude` scales the resulting curvature uniformly.

## Primary flow and Dean secondary

The **primary** flow is built from the curvature map and produces a
rotation-like field whose curl is proportional to local curvature.

The **Dean-like secondary** flow is a divergence-free perturbation built from
the skew gradient of curvature:

```text
secondary = (-d kappa / dy, d kappa / dx) * dean_secondary_strength
```

In a real microchannel, Dean vortices are 3D pairs in the channel
cross-section whose strength scales with curvature (Dean number `~ kappa`).
The 2D proxy preserves that scaling and is divergence-free up to
discretization. On a constant-curvature field it vanishes identically (see
`test_dean_secondary_zero_on_constant_curvature`).

## Shear tensor, shear magnitude, shear gradient

For a 2D incompressible flow the symmetric part of the strain-rate tensor is
dominated by the off-diagonal components. We store `(du/dy, dv/dx)` as a
`[H, W, 2]` field. Helpers:

- `shear_magnitude(shear)` returns the scalar `sqrt(s_xy^2 + s_yx^2)`.
- `shear_grad` is the spatial gradient of that scalar magnitude; it is what
  the inertial-lift proxy reads to localize "shear ridges".

## Inertial-lift proxy + wall repulsion

```text
lift  ~  skew(shear) * U * (1 + |kappa|) * (eps + |shear|) + wall_repulsion
```

`skew(shear) = (s_yx, -s_xy)` keeps the proxy roughly perpendicular to the
local shear. The `(1 + |kappa|)` factor strengthens lift in high-curvature
regions, the `(eps + |shear|)` term enforces the shear-gradient origin of
inertial lift in real microfluidics, and `U = |flow|` recovers the `U^2`-like
amplitude (one `U` from `skew(shear)` which scales with the velocity scale,
one explicit `U` here).

When `toroidal=False`, an additional **wall-repulsion** term is added that
points away from each boundary with a tent-shaped falloff active only within
the boundary strip. This mimics the lift-equilibrium balance that pushes
particles to non-zero equilibrium distances from channel walls.

## Enrichment / margination

```text
enrichment = tanh(
    sum(resources) - sum(hazards) + 0.25 * |lift|
    - (0.15 if occupied else 0)
)
```

The lift-magnitude bonus models how inertial lift concentrates particles into
focusing streaks. The optional occupancy penalty captures the fact that
crowded cells lose margination efficiency. Both terms are smooth and bounded
via `tanh`.

## Flow-advected concentration

`concentration` is updated each step by

1. five-point stencil diffusion + multiplicative decay (`diffuse_decay`),
2. semi-Lagrangian bilinear advection by `flow` with timestep
   `flow_advection_strength` (`advect_field`).

Setting `flow_advection_strength=0` disables advection so that agents
experience a purely diffusing scalar.

## Smooth temporal drift

When `drifting=True`, the flow field is updated each step by

```text
flow_t = 0.99 * rotate(flow_{t-1}, drift_rotation_rate * t) * (1 + 0.1 * sin(omega t))
         + drift_speed * N(0, I)
```

The combination of a slow rotation, a low-frequency sinusoidal modulation, and
Gaussian jitter produces a non-stationary but smooth environment. The
sinusoidal phase is keyed to `world.time`, so the drift is reproducible from a
given seed.

## Sixth-sense modalities

`sample_sixth_sense` produces a `[N, 11]` tensor per agent, concatenated as:

| slot | source         | width |
| ---- | -------------- | ----- |
| 0    | curvature      | 1     |
| 1    | shear_grad     | 2     |
| 2    | enrichment     | 1     |
| 3    | lift           | 2     |
| 4    | concentration  | 2     |
| 5    | crowd context  | 3     |

The crowd context is `(local_density, mean_flow_speed, neighbor_msg_energy)`
computed with scatter-add primitives so it remains vectorized across N.

### Sensing modes

All knobs are independent and compose:

| knob                                | effect                                                      |
| ----------------------------------- | ----------------------------------------------------------- |
| `blind`                             | return all zeros (overrides everything)                     |
| `true_sixth_sense=False`            | zero out the sixth-sense channels but keep patches          |
| `shuffle_sixth_sense`               | permute the 11 channels per step (destroys semantics)       |
| `sense_delayed`, `sense_delay_steps`| sample upstream-of-flow (stale reading proxy)               |
| `sense_multiplicative_noise_std`    | heteroskedastic multiplicative Gaussian noise               |
| `sense_dropout_frac`                | randomly zero a fraction of channels each step              |
| `WorldConfig.sensory_noise_std`     | additive Gaussian noise                                     |

`sense_delayed` works by advecting each agent's sampling position backward
along the local flow vector by `sense_delay_steps`; the deeper the agent sits
in a high-shear flow, the more "stale" its readings.

## Crowding / hemodynamic context

Three population-level quantities are derived from positions and the flow:

- `compute_density(positions, alive, h, w)` -> `[H, W]` per-cell counts.
- `crowding_context(...)` -> `[N, 3]` per-agent
  `(density-1, flow_speed, neighbor_message_energy)`.
- `occupancy_pressure(...)` -> `[N]` scalar `density * flow_speed` at the
  agent's cell. This is the proxy that drives margination penalties in
  high-shear crowded regions.

## Vectorization & differentiability

Every kernel listed above is a pure `jax.numpy` operation: no Python loops in
the hot path (`build_curvature_map` has a 4-step smoothing pass; even this is
a static `for` over `jnp` operations and is unrolled). All operators preserve
shapes exactly and have been finite-value-checked in the test suite (see
`tests/test_world_model.py`).

The current `step_world` is not wrapped in `jit` because it mutates a Python
dataclass, but every kernel inside it is `jit`-compatible and can be JITed
piecewise.
