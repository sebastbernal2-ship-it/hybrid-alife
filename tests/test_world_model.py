"""Unit tests for the deeper world / sensing substrate.

Covers:
  - Curvature regimes: shape, finiteness, regime-specific structure.
  - Flow / shear / lift: shape, finiteness, wall-repulsion sign.
  - Advection: mass approximate conservation, plus monotone drift in flow dir.
  - Diffusion: spread of a delta source.
  - Dean secondary flow: zero on a constant curvature, finite otherwise.
  - Sixth-sense modes: shape and ablation behavior, dropout, delayed sampling.
  - Crowding / occupancy pressure: density adds with agent count.
  - Drifting world: smooth temporal drift produces finite, varying flow.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.runner import load_config
from hybrid_alife.types import EmbodiedConfig, WorldConfig
from hybrid_alife.world.env import (
    _drift_flow,
    advect_field,
    build_curvature_map,
    compute_density,
    dean_secondary_flow,
    diffuse_decay,
    initialize_world,
    shear_magnitude,
    step_world,
)
from hybrid_alife.world.sensing import (
    SIXTH_SENSE_WIDTHS,
    crowding_context,
    occupancy_pressure,
    sample_sixth_sense,
    sixth_sense_dim,
)


def _world_cfg(**overrides) -> WorldConfig:
    base = load_config("configs/base.yaml").world
    return replace(base, **overrides)


def _embodied_cfg(**overrides) -> EmbodiedConfig:
    base = load_config("configs/base.yaml").embodied
    return replace(base, **overrides)


# --- curvature regimes ------------------------------------------------------


def test_curvature_regimes_all_shapes_finite():
    regimes = ["default", "straight", "spiral", "serpentine", "obstacle", "random_smooth"]
    h, w = 16, 20
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, h), jnp.linspace(-1.0, 1.0, w), indexing="ij"
    )
    for r in regimes:
        cfg = _world_cfg(curvature_regime=r, height=h, width=w)
        curv = build_curvature_map(cfg, xx, yy, jax.random.PRNGKey(0))
        assert curv.shape == (h, w, 1), r
        assert bool(jnp.all(jnp.isfinite(curv))), r


def test_curvature_straight_has_small_amplitude():
    cfg = _world_cfg(curvature_regime="straight")
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1.0, 1.0, cfg.height),
        jnp.linspace(-1.0, 1.0, cfg.width),
        indexing="ij",
    )
    curv = build_curvature_map(cfg, xx, yy, jax.random.PRNGKey(0))
    assert float(jnp.max(jnp.abs(curv))) <= 0.1


def test_curvature_amplitude_scales():
    yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, 16), jnp.linspace(-1, 1, 16), indexing="ij")
    cfg_a = _world_cfg(curvature_regime="serpentine", curvature_amplitude=1.0)
    cfg_b = _world_cfg(curvature_regime="serpentine", curvature_amplitude=2.5)
    a = build_curvature_map(cfg_a, xx, yy, jax.random.PRNGKey(0))
    b = build_curvature_map(cfg_b, xx, yy, jax.random.PRNGKey(0))
    np.testing.assert_allclose(np.asarray(b) / 2.5, np.asarray(a), atol=1e-5)


def test_curvature_unknown_regime_raises():
    cfg = _world_cfg(curvature_regime="not_a_regime")
    yy, xx = jnp.meshgrid(jnp.linspace(-1, 1, 8), jnp.linspace(-1, 1, 8), indexing="ij")
    try:
        build_curvature_map(cfg, xx, yy, jax.random.PRNGKey(0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


# --- flow / shear / lift ----------------------------------------------------


def test_world_fields_all_finite():
    cfg = _world_cfg()
    world = initialize_world(cfg, jax.random.PRNGKey(0))
    for f in [world.flow, world.curvature, world.shear, world.shear_grad,
              world.lift, world.enrichment, world.concentration, world.metabolites]:
        assert bool(jnp.all(jnp.isfinite(f)))


def test_shear_magnitude_shape_and_nonneg():
    cfg = _world_cfg()
    world = initialize_world(cfg, jax.random.PRNGKey(0))
    mag = shear_magnitude(world.shear)
    assert mag.shape == (cfg.height, cfg.width)
    assert float(jnp.min(mag)) >= 0.0


def test_dean_secondary_zero_on_constant_curvature():
    kappa = 0.7 * jnp.ones((8, 8, 1), dtype=jnp.float32)
    sec = dean_secondary_flow(kappa)
    # divergence-free skew-grad of constant is zero
    assert float(jnp.max(jnp.abs(sec))) < 1e-6


def test_dean_secondary_nonzero_on_varying_curvature():
    cfg = _world_cfg(curvature_regime="serpentine")
    yy, xx = jnp.meshgrid(
        jnp.linspace(-1, 1, cfg.height),
        jnp.linspace(-1, 1, cfg.width),
        indexing="ij",
    )
    curv = build_curvature_map(cfg, xx, yy, jax.random.PRNGKey(0))
    sec = dean_secondary_flow(curv)
    assert float(jnp.linalg.norm(sec)) > 0.0


def test_wall_repulsion_only_when_non_toroidal():
    cfg_t = _world_cfg(toroidal=True, wall_repulsion_strength=1.0, flow_noise_std=0.0)
    cfg_nt = _world_cfg(toroidal=False, wall_repulsion_strength=1.0, flow_noise_std=0.0)
    w_t = initialize_world(cfg_t, jax.random.PRNGKey(0))
    w_nt = initialize_world(cfg_nt, jax.random.PRNGKey(0))
    # the non-toroidal lift should have larger boundary-zone magnitudes
    border_t = float(jnp.linalg.norm(w_t.lift[0, :, :]))
    border_nt = float(jnp.linalg.norm(w_nt.lift[0, :, :]))
    assert border_nt > border_t


# --- advection / diffusion --------------------------------------------------


def test_advect_field_preserves_mean_approximately():
    h, w = 16, 16
    field = jax.random.normal(jax.random.PRNGKey(0), (h, w, 2), dtype=jnp.float32)
    flow = jnp.ones((h, w, 2), dtype=jnp.float32) * 0.5
    out = advect_field(field, flow, dt=1.0, toroidal=True)
    np.testing.assert_allclose(float(field.mean()), float(out.mean()), atol=1e-3)
    assert out.shape == field.shape


def test_advect_field_moves_blob_in_flow_direction():
    h, w = 16, 16
    field = jnp.zeros((h, w, 1), dtype=jnp.float32)
    field = field.at[8, 8, 0].set(1.0)
    flow = jnp.zeros((h, w, 2), dtype=jnp.float32)
    flow = flow.at[..., 1].set(1.0)  # advect along x
    out = advect_field(field, flow, dt=1.0, toroidal=True)
    # after advection, mass should be near column 9, not 8
    col9 = float(out[8, 9, 0])
    col8 = float(out[8, 8, 0])
    assert col9 > col8


def test_diffuse_decay_spreads_delta():
    h, w = 16, 16
    field = jnp.zeros((h, w, 1), dtype=jnp.float32).at[8, 8, 0].set(1.0)
    out = diffuse_decay(field, decay=1.0, diffusion=0.2, toroidal=True)
    # neighbors should now be > 0
    assert float(out[7, 8, 0]) > 0.0
    assert float(out[9, 8, 0]) > 0.0
    assert float(out[8, 8, 0]) < 1.0


def test_step_world_advects_concentration():
    cfg = _world_cfg(flow_advection_strength=0.8, flow_noise_std=0.0)
    world = initialize_world(cfg, jax.random.PRNGKey(0))
    # seed a delta and step
    conc = jnp.zeros_like(world.concentration).at[8, 8, 0].set(5.0)
    world.concentration = conc
    from hybrid_alife.types import SimState
    state = SimState(
        generation=0, step=0, rng=jax.random.PRNGKey(1),
        world=world, embodied=None, avida=None
    )
    state = step_world(state, cfg)
    # total mass should drop (decay) but should not blow up; finite everywhere
    assert bool(jnp.all(jnp.isfinite(state.world.concentration)))
    assert float(state.world.concentration.sum()) < 5.0 * cfg.height * cfg.width


# --- drifting world ---------------------------------------------------------


def test_drift_flow_changes_field_over_time():
    cfg = _world_cfg(drifting=True, drift_speed=0.05, drift_rotation_rate=0.2)
    world = initialize_world(cfg, jax.random.PRNGKey(0))
    f0 = world.flow
    f1 = _drift_flow(f0, jnp.asarray(5, dtype=jnp.int32), cfg, jax.random.PRNGKey(2))
    f2 = _drift_flow(f0, jnp.asarray(50, dtype=jnp.int32), cfg, jax.random.PRNGKey(3))
    assert bool(jnp.all(jnp.isfinite(f1)))
    assert bool(jnp.all(jnp.isfinite(f2)))
    # the drifted fields should differ from the original
    assert float(jnp.linalg.norm(f2 - f0)) > 0.0


def test_static_world_flow_unchanged():
    cfg = _world_cfg(drifting=False, flow_noise_std=0.0)
    world = initialize_world(cfg, jax.random.PRNGKey(0))
    f_before = world.flow
    from hybrid_alife.types import SimState
    state = SimState(
        generation=0, step=0, rng=jax.random.PRNGKey(1),
        world=world, embodied=None, avida=None
    )
    state = step_world(state, cfg)
    np.testing.assert_allclose(np.asarray(f_before), np.asarray(state.world.flow), atol=1e-6)


# --- sixth-sense ablations --------------------------------------------------


def _setup_sense(**emb_overrides):
    cfg = load_config("configs/base.yaml")
    key = jax.random.PRNGKey(0)
    world = initialize_world(cfg.world, key)
    n = 4
    positions = jnp.array([[0.1, 0.1], [0.5, 0.5], [0.9, 0.2], [0.3, 0.8]])
    alive = jnp.ones((n,), dtype=bool)
    messages = jnp.ones((n, cfg.embodied.message_size), dtype=jnp.float32)
    emb = replace(cfg.embodied, **emb_overrides)
    return world, positions, alive, messages, cfg.world, emb, key


def test_sixth_sense_width_matches_layout_widths():
    assert sum(SIXTH_SENSE_WIDTHS) == sixth_sense_dim(_world_cfg())


def test_sense_delayed_mode_runs_and_finite():
    world, pos, alive, msgs, wc, ec, key = _setup_sense(
        sense_delayed=True, sense_delay_steps=3
    )
    sense = sample_sixth_sense(world, pos, alive, msgs, wc, ec, key)
    assert sense.shape == (pos.shape[0], sixth_sense_dim(wc))
    assert bool(jnp.all(jnp.isfinite(sense)))


def test_sense_multiplicative_noise_increases_variance():
    world, pos, alive, msgs, wc, ec_clean, key = _setup_sense(
        sense_multiplicative_noise_std=0.0,
    )
    ec_noisy = replace(ec_clean, sense_multiplicative_noise_std=0.5)
    samples_clean = jnp.stack(
        [sample_sixth_sense(world, pos, alive, msgs, wc, ec_clean, jax.random.PRNGKey(i))
         for i in range(8)]
    )
    samples_noisy = jnp.stack(
        [sample_sixth_sense(world, pos, alive, msgs, wc, ec_noisy, jax.random.PRNGKey(i))
         for i in range(8)]
    )
    # variance across draws should be larger when multiplicative noise is on
    assert float(samples_noisy.var()) > float(samples_clean.var())


def test_sense_dropout_zeros_some_channels_on_average():
    world, pos, alive, msgs, wc, ec, key = _setup_sense(sense_dropout_frac=0.6)
    # zero out world sensory noise to make dropout effect visible
    wc_clean = replace(wc, sensory_noise_std=0.0)
    sense = sample_sixth_sense(world, pos, alive, msgs, wc_clean, ec, key)
    zero_cols = int(jnp.sum(jnp.all(sense == 0.0, axis=0)))
    assert zero_cols > 0


def test_sense_blind_overrides_everything():
    world, pos, alive, msgs, wc, ec, key = _setup_sense(
        blind=True,
        sense_delayed=True,
        sense_multiplicative_noise_std=0.5,
        sense_dropout_frac=0.3,
    )
    sense = sample_sixth_sense(world, pos, alive, msgs, wc, ec, key)
    assert bool(jnp.all(sense == 0.0))


def test_sense_shuffle_preserves_total_energy_approximately():
    world, pos, alive, msgs, wc, ec, key = _setup_sense()
    wc = replace(wc, sensory_noise_std=0.0)
    ec_shuf = replace(ec, shuffle_sixth_sense=True)
    s = sample_sixth_sense(world, pos, alive, msgs, wc, ec, key)
    s_sh = sample_sixth_sense(world, pos, alive, msgs, wc, ec_shuf, key)
    np.testing.assert_allclose(
        float(jnp.sum(s**2, axis=-1).sum()),
        float(jnp.sum(s_sh**2, axis=-1).sum()),
        atol=1e-4,
    )


# --- crowding / occupancy pressure ------------------------------------------


def test_compute_density_counts_agents_in_cell():
    pos = jnp.array([[0.05, 0.05], [0.04, 0.04], [0.9, 0.9]])
    alive = jnp.array([True, True, True])
    d = compute_density(pos, alive, 10, 10)
    assert float(d[0, 0]) == 2.0
    assert float(d[9, 9]) == 1.0


def test_crowding_context_neighbor_count():
    n = 4
    pos = jnp.array([[0.05, 0.05], [0.04, 0.04], [0.9, 0.9], [0.5, 0.5]])
    alive = jnp.ones((n,), dtype=bool)
    msgs = jnp.zeros((n, 4), dtype=jnp.float32)
    flow = jnp.zeros((10, 10, 2), dtype=jnp.float32)
    crowd = crowding_context(pos, alive, msgs, flow, 10, 10)
    # agents 0 and 1 share a cell -> each sees one neighbor
    assert float(crowd[0, 0]) == 1.0
    assert float(crowd[1, 0]) == 1.0
    assert float(crowd[2, 0]) == 0.0


def test_occupancy_pressure_scales_with_density_and_flow():
    n = 3
    pos = jnp.array([[0.1, 0.1], [0.1, 0.1], [0.5, 0.5]])
    alive = jnp.ones((n,), dtype=bool)
    flow = jnp.ones((10, 10, 2), dtype=jnp.float32)  # uniform speed = sqrt(2)
    p = occupancy_pressure(pos, alive, flow, 10, 10)
    # agent 2 is alone, agents 0 and 1 share a cell -> higher pressure
    assert float(p[0]) > float(p[2])
    assert float(p[1]) > float(p[2])
