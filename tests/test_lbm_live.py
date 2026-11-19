from __future__ import annotations

import pickle
from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, load_config, step_sim
from hybrid_alife.physics import lbm
from hybrid_alife.physics.lbm import LiveLBM, agent_reaction_forces
from hybrid_alife.replay.checkpoint import load_checkpoint, restore_sim_state, save_checkpoint
from hybrid_alife.world.env import step_world


def _box(h=12, w=10) -> LiveLBM:
    """Closed periodic fluid at rest, no walls, no body force."""
    mask = np.zeros((w, h), np.uint8)
    f = np.zeros((w, h, 9))
    lbm._load().lbm_init(w, h, 1.0, np.zeros((w, h)), np.zeros((w, h)), f)
    z = np.zeros((w, h))
    return LiveLBM(f, mask, 0.1, 0.0, 0.05, z, z, np.ones((w, h)), np.zeros((h, w, 2)))


def test_live_channel_reaches_the_steady_solve_and_poiseuille():
    h, w, nu, peak = 8, 18, 0.1, 0.05
    live = LiveLBM.channel(h, w, nu=nu, peak_speed=peak, warm_steps=6000)
    # World (H, W) -> LBM frame [W, H]; walls on LBM rows 0 and -1 (world columns).
    mask = lbm.channel_mask(w, h)
    u, v, _ = lbm.solve(mask, nu, (live.body_force, 0.0), 12000)
    np.testing.assert_allclose(live.u, u, atol=2e-4)
    np.testing.assert_allclose(live.v, v, atol=2e-4)
    height = w - 2
    y = np.arange(1, w - 1) - 0.5
    analytic = live.body_force / (2 * nu) * y * (height - y)
    assert np.max(np.abs(live.u[1:-1, h // 2] - analytic)) / analytic.max() < 0.03
    assert abs(live.u.max() - peak) / peak < 0.03
    # World-frame flow: component 0 carries the channel flow, walls are columns.
    flow = live.flow()
    assert flow.shape == (h, w, 2) and abs(flow[..., 0].max() - 1.0) < 0.03
    assert np.all(flow[:, 0] == 0) and np.all(flow[:, -1] == 0)


def test_force_pulse_changes_momentum_by_exactly_the_impulse():
    box = _box()
    force = np.zeros((12, 10, 2))
    force[3, 4] = (0.02, -0.01)
    force[8, 2] = (0.005, 0.03)
    impulse = force.sum(axis=(0, 1))
    p0 = box.momentum()
    pulsed = box.advance(1, force=force)
    np.testing.assert_allclose(pulsed.momentum() - p0, impulse, atol=1e-12)
    # Periodic and force-free afterwards: momentum is conserved while the pulse spreads.
    later = pulsed.advance(40, force=np.zeros_like(force))
    np.testing.assert_allclose(later.momentum() - p0, impulse, atol=1e-10)
    # k steps of constant force give k times the impulse.
    held = box.advance(7, force=force)
    np.testing.assert_allclose(held.momentum() - p0, 7 * impulse, atol=1e-10)


def test_agent_reaction_obeys_newtons_third_law_and_drives_the_fluid():
    h, w, coeff, radius = 12, 10, 2.0, 1.5
    swim = np.array([[0.04, 0.0], [0.0, -0.02]])
    positions = np.array([[0.3, 0.3], [0.7, 0.8]])
    solid = np.zeros((h, w), np.uint8)
    force = agent_reaction_forces(
        positions, swim, np.array([radius, 1.0]), np.array([True, True]), solid, coeff)
    drag_on_agents = -coeff * np.array([radius, 1.0])[:, None] * swim  # Stokes drag
    np.testing.assert_allclose(force.sum(axis=(0, 1)), -drag_on_agents.sum(axis=0), atol=1e-12)
    assert force[3, 3, 0] > 0 and force[3, 3, 1] == 0  # pushes the fluid the way it swims
    assert force[8, 8, 1] < 0
    # Fluid at rest, constant forcing: total momentum grows by exactly steps * force...
    box = _box(h, w)
    driven = box.advance(25, force=force)
    np.testing.assert_allclose(
        driven.momentum() - box.momentum(), 25 * force.sum(axis=(0, 1)), atol=1e-9)
    # ...and the fluid at the swimmer's cell now moves with it (sign of the wake).
    assert driven.velocity()[3, 3, 0] > 0
    # Dead agents and solid cells deposit nothing; slip is clipped for stability.
    none = agent_reaction_forces(
        positions, swim, np.ones(2), np.array([False, False]), solid, coeff)
    assert not none.any()
    fast = agent_reaction_forces(
        positions[:1], np.array([[5.0, 0.0]]), np.ones(1), np.array([True]), solid, 1.0)
    assert fast[3, 3, 0] == lbm.MAX_RELATIVE_SPEED


def test_live_state_pickles_and_does_not_alias():
    live = LiveLBM.channel(8, 14, warm_steps=200)
    clone = pickle.loads(pickle.dumps(live))
    np.testing.assert_array_equal(clone.advance(5).f, live.advance(5).f)
    before = live.f.copy()
    live.advance(5)
    np.testing.assert_array_equal(live.f, before)  # advance never mutates its input


def _small_config(drag: float):
    cfg = load_config("configs/ablation_lbm_live.yaml")
    world = replace(cfg.world, width=14, height=14, lbm_agent_drag=drag, curvature_regime="default")
    emb = replace(cfg.embodied, population_size=8)
    return replace(cfg, world=world, embodied=emb, avida=replace(cfg.avida, enabled=False))


def test_step_world_refreshes_flow_without_nans_for_200_steps():
    cfg = _small_config(0.0)
    state = initialize_sim(cfg)
    assert state.lbm is not None
    first = np.asarray(state.world.flow).copy()
    for _ in range(200):
        state = step_world(state, cfg.world)
        assert np.isfinite(np.asarray(state.world.flow)).all()
    flow = np.asarray(state.world.flow)
    assert abs(flow[..., 0].max() - 1.0) < 0.05 and np.all(flow[:, 0] == 0)
    np.testing.assert_allclose(flow, first, atol=0.05)  # steady channel stays steady
    assert state.lbm.f.shape == (14, 14, 9)


def test_agents_perturb_the_live_flow_only_when_coupled():
    flows = {}
    for drag in (0.0, 2.0):
        cfg = _small_config(drag)
        state = initialize_sim(cfg)
        counter = 0
        for _ in range(30):
            state, counter = step_sim(state, cfg, counter)
        assert np.isfinite(np.asarray(state.world.flow)).all()
        flows[drag] = np.asarray(state.world.flow)
    assert np.abs(flows[2.0] - flows[0.0]).max() > 1e-4
    # With drag 0 the fluid stays the steady channel; with drag it was measurably disturbed.
    steady = np.asarray(initialize_sim(_small_config(0.0)).world.flow)
    assert np.abs(flows[0.0] - steady).max() < 0.05


def test_checkpoint_round_trips_live_fluid(tmp_path):
    cfg = _small_config(1.0)
    state = initialize_sim(cfg)
    counter = 0
    for _ in range(5):
        state, counter = step_sim(state, cfg, counter)
    path = save_checkpoint(tmp_path / "c.pkl", state, {})
    restored = restore_sim_state(load_checkpoint(path))
    np.testing.assert_array_equal(restored.lbm.f, state.lbm.f)
    np.testing.assert_array_equal(
        restored.lbm.advance(3).f, state.lbm.advance(3).f)


def test_legacy_flow_solvers_carry_no_live_state():
    cfg = _small_config(0.0)
    for solver in ("proxy", "lbm"):
        legacy = replace(cfg, world=replace(cfg.world, flow_solver=solver))
        state = initialize_sim(legacy)
        assert state.lbm is None
        before = np.asarray(state.world.flow).copy()
        state = step_world(state, legacy.world)
        np.testing.assert_array_equal(np.asarray(state.world.flow), before)  # static as before
    _ = jnp


def test_live_lbm_rejects_bad_force_shapes_and_supersonic_flow():
    import pytest

    fluid = LiveLBM.channel(12, 12, warm_steps=10)
    with pytest.raises(ValueError, match="shape"):
        fluid.advance(1, force=np.zeros((12, 11, 2)))
    with pytest.raises(ValueError, match="shape"):
        fluid.with_pending_force(np.zeros((3, 3, 2)))
    huge = np.zeros((12, 12, 2))
    huge[6, 6] = 5.0
    with pytest.raises(ValueError, match="low-Mach"):
        fluid.advance(5, force=huge)
    # a missing pending force means no extra forcing, not a crash
    bare = LiveLBM.channel(12, 12, warm_steps=1)
    bare.pending_force = None
    assert bare.advance(1).u.shape == bare.u.shape


def test_world_config_validates_live_lbm_parameters():
    import pytest

    from hybrid_alife.types import WorldConfig

    kw = dict(width=12, height=12, toroidal=False, resource_channels=1, hazard_channels=1,
              max_agents=1, max_digital_organisms=1, flow_noise_std=0.0,
              concentration_decay=0.9, concentration_diffusion=0.05, flow_solver="lbm_live")
    WorldConfig(**kw)  # defaults are valid
    for bad, match in (
        ({"lbm_viscosity": 0.0}, "lbm_viscosity"),
        ({"lbm_peak_speed": 0.5}, "lbm_peak_speed"),
        ({"lbm_peak_speed": 0.0}, "lbm_peak_speed"),
        ({"lbm_substeps": 0}, "lbm_substeps"),
        ({"lbm_agent_drag": -1.0}, "lbm_agent_drag"),
        ({"lbm_agent_coupling": "quadrupole"}, "lbm_agent_coupling"),
        ({"drifting": True}, "ignores drifting"),
        ({"flow_noise_std": 0.03}, "ignores drifting"),
    ):
        with pytest.raises(ValueError, match=match):
            WorldConfig(**{**kw, **bad})
    with pytest.raises(ValueError, match="flow_solver"):
        WorldConfig(**{**kw, "flow_solver": "lbn"})
