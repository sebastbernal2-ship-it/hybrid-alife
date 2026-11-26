"""Live coupling with obstacle and serpentine walls: momentum bookkeeping and solid cells."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from hybrid_alife.experiments.runner import agent_force_field, initialize_sim, load_config, step_sim
from hybrid_alife.physics import lbm
from hybrid_alife.physics.lbm import (
    LiveLBM,
    agent_dipole_forces,
    agent_reaction_forces,
    redistribute_solid_forces,
    serpentine_mask,
)

H, W = 16, 18  # world


def _mask(geometry: str) -> np.ndarray:
    """Solver frame [W, H]."""
    if geometry == "serpentine":
        return serpentine_mask(W, H, waves=1, amplitude=0.15)
    return lbm.channel_mask(W, H, obstacle_radius=3.0)


def _wall_momentum(after: LiveLBM) -> np.ndarray:
    """Momentum the walls removed in the last step, read from the bounced populations.

    Halfway bounce-back returns the population sent from fluid cell i along q into a wall as
    population opposite(q) at i, so each link takes 2 * value * c_q from the fluid.
    """
    solid = after.mask != 0
    opposite = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    out = np.zeros(2)
    for q in range(1, 9):
        cx, cy = lbm._CX[q], lbm._CY[q]
        # solid[y + cy, x + cx], periodic; the solver array is [y, x].
        link = ~solid & np.roll(solid, (-cy, -cx), axis=(0, 1))
        out += 2.0 * after.f[..., opposite[q]][link].sum() * np.array([cx, cy])
    return out


def _agents(world_solid: np.ndarray):
    """Two agents on fluid cells and two on solid cells (one at the wall, one other)."""
    solid, fluid = np.argwhere(world_solid != 0), np.argwhere(world_solid == 0)
    cells = np.array([fluid[3], fluid[len(fluid) // 2], solid[0], solid[len(solid) // 2]])
    positions = (cells + 0.5) / np.array(world_solid.shape)
    swim = np.array([[0.03, 0.01], [-0.02, 0.04], [0.05, -0.02], [0.01, 0.06]])
    return positions, swim, np.ones(4), np.ones(4, bool)


@pytest.mark.parametrize("geometry", ["obstacle", "serpentine"])
def test_monopole_solid_force_is_dropped_by_default_and_kept_when_redistributed(geometry):
    solid = _mask(geometry).T
    args = (*_agents(solid), solid, 2.0)
    intended = (2.0 * args[1]).sum(axis=0)
    dropped = agent_reaction_forces(*args)
    kept = agent_reaction_forces(*args, solid_policy="redistribute")
    np.testing.assert_allclose(dropped.sum(axis=(0, 1)), (2.0 * args[1][:2]).sum(axis=0))
    np.testing.assert_allclose(kept.sum(axis=(0, 1)), intended, atol=1e-15)
    assert np.all(kept[solid != 0] == 0) and np.all(dropped[solid != 0] == 0)


@pytest.mark.parametrize("geometry", ["obstacle", "serpentine"])
def test_momentum_budget_closes_with_walls(geometry):
    """fluid momentum change = body force + agent force - momentum taken by the walls."""
    mask = _mask(geometry)
    fluid = LiveLBM.channel(H, W, mask=mask, warm_steps=300)
    force = agent_reaction_forces(
        *_agents(mask.T), mask.T, 2.0, solid_policy="redistribute"
    )
    for _ in range(5):
        rho = fluid.f.sum(axis=-1)[mask == 0].sum()
        after = fluid.advance(1, force=force)
        applied = force.sum(axis=(0, 1)) + np.array([fluid.body_force * rho, 0.0])
        np.testing.assert_allclose(
            after.momentum() - fluid.momentum(), applied - _wall_momentum(after), atol=1e-11
        )
        # Mass is conserved in the fluid.
        np.testing.assert_allclose(after.f[mask == 0].sum(), rho, atol=1e-9)
        fluid = after


def test_the_kernel_itself_ignores_force_on_solid_cells():
    mask = _mask("obstacle")
    fluid = LiveLBM.channel(H, W, mask=mask, warm_steps=50)
    on_solid = np.zeros((H, W, 2))
    on_solid[mask.T != 0] = 0.01
    np.testing.assert_array_equal(
        fluid.advance(3, force=on_solid).f, fluid.advance(3, force=np.zeros_like(on_solid)).f
    )


@pytest.mark.parametrize("geometry", ["obstacle", "serpentine"])
def test_redistributed_dipole_keeps_zero_net_force_and_stays_in_the_fluid(geometry):
    solid = _mask(geometry).T
    positions, swim, radii, alive = _agents(solid)
    force = agent_dipole_forces(
        positions, swim, radii, alive, solid, 2.0, 2.0, True, solid_policy="redistribute"
    )
    np.testing.assert_allclose(force.sum(axis=(0, 1)), 0.0, atol=1e-15)
    assert np.all(force[solid != 0] == 0) and np.any(force != 0)


def test_redistribution_reaches_past_the_first_ring_and_leaves_fluid_forces_alone():
    solid = np.zeros((9, 9), np.uint8)
    solid[2:7, 2:7] = 1
    force = np.zeros((9, 9, 2))
    force[4, 4] = (1.6, 0.0)  # centre: first fluid ring is at distance 3
    force[0, 0] = (0.0, 1.0)
    out = redistribute_solid_forces(force, solid)
    np.testing.assert_allclose(out.sum(axis=(0, 1)), (1.6, 1.0), atol=1e-14)
    np.testing.assert_array_equal(out[0, 0], (0.0, 1.0))
    assert np.all(out[solid != 0] == 0) and out[1, 4, 0] > 0 and out[0, 4, 0] == 0
    with pytest.raises(ValueError, match="no fluid"):
        redistribute_solid_forces(force, np.ones_like(solid))


def test_serpentine_mask_is_a_connected_meandering_channel():
    mask = _mask("serpentine")
    assert np.all(mask[0] == 1) and np.all(mask[-1] == 1)
    open_rows = [np.flatnonzero(mask[:, x] == 0) for x in range(mask.shape[1])]
    assert all(len(r) >= 3 for r in open_rows)
    for a, b in zip(open_rows, open_rows[1:] + open_rows[:1], strict=True):
        assert np.intersect1d(a, b).size  # connected, including across the periodic seam
    assert len({tuple(r) for r in open_rows}) > 2
    with pytest.raises(ValueError, match="whole number"):
        serpentine_mask(16, 18, waves=1.5)
    with pytest.raises(ValueError, match="half-width"):
        serpentine_mask(8, 18, amplitude=0.3)
    with pytest.raises(ValueError, match="mask must have shape"):
        LiveLBM.channel(H, W, mask=mask.T)


def _config(**world):
    cfg = load_config("configs/ablation_lbm_live.yaml")
    base = replace(
        cfg.world, width=W, height=H, curvature_regime="obstacle", lbm_agent_drag=2.0
    )
    return replace(
        cfg, world=replace(base, **world),
        embodied=replace(cfg.embodied, population_size=8), avida=replace(cfg.avida, enabled=False),
    )


@pytest.mark.parametrize("geometry", ["auto", "serpentine"])
@pytest.mark.parametrize("coupling", ["monopole", "dipole"])
def test_coupled_run_with_walls_is_finite_and_policy_reaches_the_runner(geometry, coupling):
    cfg = _config(lbm_geometry=geometry, lbm_agent_coupling=coupling,
                  lbm_solid_force="redistribute")
    state = initialize_sim(cfg)
    solid = state.lbm.mask.T != 0
    # Park every agent on a wall: drop would deposit nothing at all.
    walls = np.argwhere(solid)[: state.embodied.alive.shape[0]]
    old = (walls + 0.5) / np.array([H, W])
    state.embodied.positions = old + np.array([0.02, 0.01])
    kept = agent_force_field(state, cfg, old)
    lost = agent_force_field(state, replace(cfg, world=replace(cfg.world, lbm_solid_force="drop")),
                             old)
    assert np.all(kept[solid] == 0)
    if coupling == "monopole":
        assert np.any(kept != 0) and not np.any(lost != 0)
    counter = 0
    for _ in range(20):
        state, counter = step_sim(state, cfg, counter)
    assert np.isfinite(np.asarray(state.world.flow)).all()
    assert np.all(np.asarray(state.world.flow)[solid] == 0)


def test_defaults_keep_the_recorded_behaviour_and_bad_values_are_rejected():
    cfg = load_config("configs/ablation_lbm_live.yaml")
    assert (cfg.world.lbm_geometry, cfg.world.lbm_solid_force) == ("auto", "drop")
    for bad, match in (
        ({"lbm_geometry": "spiral"}, "lbm_geometry"),
        ({"lbm_solid_force": "bounce"}, "lbm_solid_force"),
        ({"lbm_geometry": "serpentine", "lbm_box": "periodic"}, "serpentine"),
    ):
        with pytest.raises(ValueError, match=match):
            replace(cfg.world, **bad)
