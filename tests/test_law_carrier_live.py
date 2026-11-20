from __future__ import annotations

from dataclasses import replace

import numpy as np

from hybrid_alife.experiments.latent_law_embodied import (
    _impose_flow,
    embodied_law_config,
)
from hybrid_alife.experiments.runner import initialize_sim, step_sim
from hybrid_alife.physics.lbm import LiveLBM

SIZE = 14
MAGNITUDE = 0.2
DIRECTIONS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0),
              (0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071))


def _config(live: bool, drag: float = 0.0, coupling: str = "monopole", relax: float = 20.0):
    base = embodied_law_config(
        seed=0, generations=1, population=6, steps_per_generation=1, size=SIZE,
        proprioception=True, unit_flow_features=True, flow_head=True)
    if not live:
        return base
    world = replace(
        base.world, flow_solver="lbm_live", lbm_box="periodic", lbm_peak_speed=0.25,
        lbm_law_relax_steps=relax, lbm_agent_drag=drag, lbm_agent_coupling=coupling)
    return replace(base, world=world)


def test_live_carrier_flow_follows_law_and_settles_without_agents():
    cfg = _config(True)
    cfg = replace(cfg, embodied=replace(cfg.embodied, initial_alive_fraction=0.0))
    for direction in DIRECTIONS:
        unit = np.asarray(direction) / np.linalg.norm(direction)
        for sign in (1.0, -1.0):  # sign -1 is the reversed-law control
            state = initialize_sim(cfg)
            state = _impose_flow(state, tuple(sign * unit), MAGNITUDE)
            first = np.asarray(state.world.flow).copy()
            for _ in range(60):  # settled: 60 world steps = 600 LBM steps change nothing
                state.lbm = state.lbm.advance(cfg.world.lbm_substeps)
            flow = np.asarray(state.lbm.flow())
            np.testing.assert_allclose(flow, first, atol=1e-6)
            cos = flow.reshape(-1, 2) @ (sign * unit) / np.linalg.norm(flow, axis=-1).reshape(-1)
            assert cos.min() > 0.99
            np.testing.assert_allclose(np.linalg.norm(flow, axis=-1), MAGNITUDE, atol=1e-6)


def test_uncoupled_live_carrier_equals_legacy_uniform_flow_to_float32():
    legacy = initialize_sim(_config(False))
    live = initialize_sim(_config(True))
    for direction in DIRECTIONS:
        a = np.asarray(_impose_flow(legacy, direction, MAGNITUDE).world.flow)
        b = np.asarray(_impose_flow(live, direction, MAGNITUDE).world.flow)
        np.testing.assert_allclose(a, b, atol=2e-7)
    assert legacy.lbm is None  # legacy path carries no fluid


def test_reimposing_same_law_is_a_noop_but_new_direction_restarts_fluid():
    state = initialize_sim(_config(True))
    state = _impose_flow(state, (1.0, 0.0), MAGNITUDE)
    state.lbm = state.lbm.with_pending_force(np.ones((SIZE, SIZE, 2)) * 1e-4).advance(10)
    perturbed = state.lbm
    state = _impose_flow(state, (1.0, 0.0), MAGNITUDE)
    assert state.lbm is perturbed  # unchanged law keeps the evolved fluid
    state = _impose_flow(state, (0.0, 1.0), MAGNITUDE)
    assert state.lbm is not perturbed
    np.testing.assert_allclose(state.lbm.mean_velocity(), [0.0, MAGNITUDE * 0.25], atol=1e-12)


def test_mean_flow_controller_holds_law_against_a_net_force():
    box = LiveLBM.uniform(SIZE, SIZE, nu=0.1, peak_speed=0.25, velocity=(0.05, 0.0), relax_steps=20)
    free = LiveLBM.uniform(SIZE, SIZE, nu=0.1, peak_speed=0.25, velocity=(0.05, 0.0))
    force = np.zeros((SIZE, SIZE, 2))
    force[..., 0] = 1e-4
    held, drifting = box.with_pending_force(force), free.with_pending_force(force)
    for _ in range(40):
        held, drifting = held.advance(10), drifting.advance(10)
    offset = held.mean_velocity()[0] - 0.05
    assert abs(offset - 1e-4 * 20) < 0.1 * 1e-4 * 20  # steady offset = force * relax time
    assert drifting.mean_velocity()[0] - 0.05 > 0.03  # no controller: accelerates freely


def test_dipole_keeps_mean_flow_where_monopole_shifts_it():
    shifts = {}
    for coupling in ("monopole", "dipole"):
        cfg = _config(True, drag=0.5, coupling=coupling, relax=0.0)  # no controller
        state = _impose_flow(initialize_sim(cfg), (1.0, 0.0), MAGNITUDE)
        counter = cfg.embodied.population_size
        for _ in range(6):
            state, counter = step_sim(state, cfg, counter, allow_reproduction=False)
            assert np.isfinite(np.asarray(state.world.flow)).all()
        shifts[coupling] = abs(state.lbm.mean_velocity() - state.lbm.target).max()
    # Dipole net force is zero up to the density weighting of the Guo force (rho varies at
    # the 1e-6 level near agents), so compare orders of magnitude, not exact zero.
    assert shifts["dipole"] < 1e-3 * shifts["monopole"]
    assert shifts["monopole"] > 1e-4
