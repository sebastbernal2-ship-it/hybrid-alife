from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from hybrid_alife.physics import ib, lbm
from hybrid_alife.physics.units import UnitMap, oscillatory_reynolds

rng = np.random.default_rng(0)


def _swimmer_sim(**kw):
    s = ib.Structure.from_bodies([ib.swimmer((40.0, 32.21), **kw)], ramp_steps=1000)
    return ib.ImmersedSim.create(s, 64, 96)


def test_spreading_preserves_total_force_and_torque():
    X = rng.uniform(8, 24, (20, 2))
    F = rng.normal(size=(20, 2))
    grid = ib.spread((32, 32), X, F)
    yy, xx = np.mgrid[0:32, 0:32]
    np.testing.assert_allclose(grid.sum((0, 1)), F.sum(0), atol=1e-12)
    torque_grid = np.sum(xx * grid[..., 1] - yy * grid[..., 0])
    torque = np.sum(X[:, 0] * F[:, 1] - X[:, 1] * F[:, 0])
    np.testing.assert_allclose(torque_grid, torque, atol=1e-10)


def test_spread_rejects_non_finite_points():
    with pytest.raises(ValueError, match="finite"):
        ib.spread((8, 8), np.full((1, 2), np.nan), np.zeros((1, 2)))


def test_internal_forces_are_the_energy_gradient_and_force_and_torque_free():
    s = ib.Structure.from_bodies([ib.swimmer((20.0, 20.0), heading=0.7)], ramp_steps=100)
    X = s.points + rng.normal(0, 0.15, s.points.shape)
    _, F = s.forces(X, 777.0)
    np.testing.assert_allclose(F.sum(0), 0, atol=1e-13)
    assert abs(np.sum(X[:, 0] * F[:, 1] - X[:, 1] * F[:, 0])) < 1e-12
    h = 1e-6
    for p, d in [(0, 0), (1, 1), (len(X) - 1, 0), (len(X) - 5, 1)]:
        plus, minus = X.copy(), X.copy()
        plus[p, d] += h
        minus[p, d] -= h
        grad = (s.forces(plus, 777.0)[0] - s.forces(minus, 777.0)[0]) / (2 * h)
        assert F[p, d] == pytest.approx(-grad, abs=1e-8)


def test_swimmer_is_force_free_conserves_momentum_and_swims_head_first():
    sim = _swimmer_sim()
    start = sim.centroid()
    out = sim.advance(8000)
    rep = out.report
    assert rep["max_force_scale"] > 0.01  # the body really is being actuated
    assert rep["max_net_force"] < 1e-14 and rep["max_net_torque"] < 1e-12
    np.testing.assert_allclose(out.fluid_momentum(), sim.fluid_momentum(), atol=1e-10)
    assert out.centroid()[0] - start[0] > 0.4  # wave runs tailward, body goes +x
    np.testing.assert_array_equal(sim.X, sim.structure.points)  # advance does not mutate
    # Splitting a run is bit-identical to running it in one call.
    np.testing.assert_array_equal(sim.advance(3000).advance(5000).f, out.f)


def test_unactuated_swimmer_stays_put():
    sim = _swimmer_sim(wave_amplitude=0.0)
    out = sim.advance(2000)
    np.testing.assert_allclose(out.X, sim.X, atol=1e-9)


def test_cylinder_drag_matches_sangani_acrivos_and_converges():
    """Square array of cylinders: F/(mu U) = 4 pi / D(c), Sangani & Acrivos (1982)."""
    c, nu, errors = 0.05, 1 / 6, []
    kref = 4 * np.pi / (-0.5 * np.log(c) - 0.738 + c - 0.887 * c**2 + 2.038 * c**3)
    for n, steps in ((32, 5000), (64, 20000)):
        radius = n * np.sqrt(c / np.pi)
        g = 0.002 * nu * kref / n**2
        s = ib.Structure.from_bodies(
            [ib.cylinder((n / 2 + 0.13, n / 2 + 0.29), radius, spacing=1.0)], tether_k=0.05)
        sim = ib.ImmersedSim.create(s, n, n, nu=nu, body_force=(g, 0.0)).advance(steps)
        drag = -sim.report["tether_fx"]
        assert drag == pytest.approx(g * sim.rho.sum(), rel=2e-3)  # momentum balance
        errors.append(drag / (nu * sim.u.mean()) / kref - 1)
    # The delta kernel makes the cylinder about half a cell fatter: drag is high.
    assert 0.15 < errors[0] < 0.22 and 0.06 < errors[1] < 0.11
    assert errors[1] < 0.55 * errors[0]


def test_taylor_sheet_swims_against_the_wave_within_the_resolved_error():
    """Taylor (1951) with Tuck's finite-inertia factor; 48 cells per wavelength."""
    n, nu, period = 48, 1 / 6, 9000
    k, omega = 2 * np.pi / n, 2 * np.pi / period
    s = ib.Structure.from_bodies(
        [ib.sheet(n, n / 2 + 0.17, 0.2 / k, period=period, bend_k=3.0)], ramp_steps=period)
    sim = ib.ImmersedSim.create(s, n, n, nu=nu).advance(2 * period)
    x0 = sim.X[:, 0].mean()
    far, amps = [], []
    for _ in range(80):
        sim = sim.advance(period // 40)
        y = sim.X[:, 1] - sim.X[:, 1].mean()
        amps.append(2 * abs(np.mean(y * np.exp(-1j * k * sim.X[:, 0]))))
        far.append(sim.u[int(round(sim.X[:, 1].mean() + n / 2)) % n].mean())
    speed = (sim.X[:, 0].mean() - x0) / (2 * period) - np.mean(far)
    amp = np.mean(amps)
    f = np.sqrt((1 + np.sqrt(1 + oscillatory_reynolds(omega, k, nu) ** 2)) / 2)
    theory = 0.5 * omega * k * amp**2 * (1 - 19 / 16 * (amp * k) ** 2) * (1 + f) / (2 * f)
    assert speed < 0
    assert -0.27 < -speed / theory - 1 < -0.17  # measured -22 %; see the docs table


def test_unstable_and_invalid_runs_raise_and_leave_state_alone():
    sim = _swimmer_sim(bend_k=500.0)
    before = sim.f.copy()
    with pytest.raises(ValueError, match="unstable"):
        sim.advance(3000)
    np.testing.assert_array_equal(sim.f, before)
    with pytest.raises(ValueError, match="invalid"):
        ib.ImmersedSim.create(sim.structure, 64, 96, nu=0.0).advance(1)
    with pytest.raises(ValueError, match="shape"):
        replace(sim, X=sim.X[:-1]).advance(1)
    with pytest.raises(ValueError, match="at least one body"):
        ib.Structure.from_bodies([])
    with pytest.raises(ValueError, match="mask"):
        ib.ImmersedSim.create(sim.structure, 64, 96, mask=np.zeros((3, 3)))
    with pytest.raises(ValueError, match="out of range"):
        ib.Structure.from_bodies([ib.Body(np.zeros((2, 2)), "bad", np.array([[0, 5]]), [1.0])])
    with pytest.raises(ValueError, match="degenerate"):
        sim.structure.forces(np.zeros_like(sim.X), 0.0)


def test_couples_to_the_live_channel_and_refuses_to_touch_walls():
    live = lbm.LiveLBM.channel(96, 40, warm_steps=3000)  # LBM frame is [40, 96]
    s = ib.Structure.from_bodies([ib.swimmer((30.0, 20.0))], ramp_steps=1000)
    sim = ib.ImmersedSim.from_live(s, live)
    # With no actuation time yet, the body is carried downstream by the live flow.
    assert sim.advance(300).centroid()[0] - sim.centroid()[0] > 5
    near = ib.Structure.from_bodies([ib.swimmer((30.0, 4.0))])
    with pytest.raises(ValueError, match="solid"):
        ib.ImmersedSim.from_live(near, live).advance(1)


def test_spatial_hash_matches_brute_force_and_conserves_momentum():
    for box, radius in (((40.0, 30.0), 1.5), ((7.0, 5.0), 2.4)):  # second: too few cells
        X = rng.uniform(-50, 90, (300, 2))
        body = rng.integers(0, 12, 300)
        F, pairs, dmin = ib.contact_forces(X, body, box, radius, 0.3)
        G, pairs_b, dmin_b = ib.contact_forces(X, body, box, radius, 0.3, brute=True)
        assert pairs == pairs_b > 0 and dmin == dmin_b
        np.testing.assert_allclose(F, G, atol=1e-13)
        np.testing.assert_allclose(F.sum(0), 0, atol=1e-12)
    # Pair forces are central, so they exert no torque (no periodic wrap involved here).
    X = rng.uniform(10, 20, (80, 2))
    F, _, _ = ib.contact_forces(X, np.arange(80), (100.0, 100.0), 2.0, 0.3)
    assert abs(np.sum(X[:, 0] * F[:, 1] - X[:, 1] * F[:, 0])) < 1e-12
    assert ib.contact_forces(X, np.zeros(80), (100.0, 100.0), 2.0, 0.3)[1] == 0  # same body
    with pytest.raises(ValueError):
        ib.contact_forces(X, np.arange(80), (3.0, 100.0), 2.0, 0.3)


def _inside(points, polygon):
    """Even-odd rule point-in-polygon."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    x, y = points[:, :1], points[:, 1:]
    straddle = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        cross = a[:, 0] + (y - a[:, 1]) / (b[:, 1] - a[:, 1]) * (b[:, 0] - a[:, 0])
    return np.sum(straddle & (x < cross), axis=1) % 2 == 1


def _head_on(contact_k):
    """Two cells pushed together by an incompressible stagnation flow (zero momentum)."""
    kw = dict(tail_length=6.0, wave_amplitude=0.0, spacing=0.9)
    bodies = [ib.swimmer((24.0, 24.2), 0.0, **kw), ib.swimmer((40.0, 24.2), np.pi, **kw)]
    s = ib.Structure.from_bodies(bodies, contact_radius=3.0, contact_k=contact_k)
    sim = ib.ImmersedSim.create(s, 48, 64)
    yy, xx = np.mgrid[0:48, 0:64]
    kx, ky = 2 * np.pi / 64, 2 * np.pi / 48
    u = 0.2 * np.sin(kx * xx) * np.cos(ky * (yy - 24))
    v = -0.2 * kx / ky * np.cos(kx * xx) * np.sin(ky * (yy - 24))
    lbm._load().lbm_init(48, 64, 1.0, np.ascontiguousarray(u), np.ascontiguousarray(v), sim.f)
    a, b = (np.flatnonzero(s.body == n) for n in (0, 1))
    p0, closest = sim.fluid_momentum(), np.inf
    for _ in range(300):
        sim = sim.advance(2)
        assert sim.report["max_net_force"] < 1e-12
        closest = min(closest, np.linalg.norm(sim.X[a, None] - sim.X[None, b], axis=-1).min())
        ring_a, ring_b = sim.X[a[1:-7]], sim.X[b[1:-7]]  # ring: skip hub and 7 tail points
        assert not _inside(ring_a, ring_b).any() and not _inside(ring_b, ring_a).any()
    np.testing.assert_allclose(sim.fluid_momentum(), p0, atol=1e-10)
    return closest


def test_head_on_collision_contact_holds_bodies_apart_and_keeps_momentum():
    without, held = _head_on(0.0), _head_on(0.3)
    assert without < 1.6  # the flow alone drives the surfaces well inside the cutoff
    assert held > 2.3  # soft contact: some overlap of the 3.0 cutoff, never a crossing


def test_state_export_schema(tmp_path):
    s = ib.Structure.from_bodies(
        [ib.swimmer((30.0, 20.0)), ib.cylinder((70.0, 40.0), 5.0)], ramp_steps=100)
    sim = ib.ImmersedSim.create(s, 64, 96)
    path = tmp_path / "state.jsonl"
    end = ib.record_states(sim, path, frames=3, steps_per_frame=20, dx_m=5e-7, dt_s=4e-8)
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    header, frames = lines[0], lines[1:]
    assert header["schema"] == ib.STATE_SCHEMA and (header["nx"], header["ny"]) == (96, 64)
    assert [b["kind"] for b in header["bodies"]] == ["swimmer", "cylinder"]
    assert header["bodies"][1]["stop"] == len(s.points)
    assert [f["step"] for f in frames] == [0, 20, 40, 60]
    np.testing.assert_allclose(frames[-1]["points"], end.X, atol=1e-5)


def test_unit_map_round_trips_and_numbers():
    units = UnitMap(dx_m=0.25e-6)
    assert units.dt_s == pytest.approx(1 / 6 * 0.25e-6**2 / 1e-6)
    assert units.velocity_to_si(units.velocity_to_lu(25e-6)) == pytest.approx(25e-6)
    assert units.reynolds(25e-6, 2e-6) == pytest.approx(5e-5)
    assert units.peclet(25e-6, 2e-6, 1e-9) == pytest.approx(0.05)
    assert units.steps_per_period(100) == pytest.approx(960_000)
    assert units.force_per_depth_to_si(1.0) == pytest.approx(
        998 * units.dx_m**3 / units.dt_s**2)
    with pytest.raises(ValueError):
        UnitMap(dx_m=0.0)


def test_default_kernel_is_four_point_and_unknown_kernels_are_rejected():
    bodies = [ib.swimmer((40.0, 32.21))]
    default = ib.ImmersedSim.create(ib.Structure.from_bodies(bodies), 64, 96).advance(200)
    four = ib.ImmersedSim.create(ib.Structure.from_bodies(bodies, kernel=4), 64, 96).advance(200)
    np.testing.assert_array_equal(default.f, four.f)
    with pytest.raises(ValueError, match="kernel"):
        ib.Structure.from_bodies(bodies, kernel=5)
    with pytest.raises(ValueError, match="invalid"):
        replace(default, structure=replace(default.structure, kernel=1)).advance(1)


@pytest.mark.parametrize("kernel", [2, 3])
def test_narrow_kernels_conserve_momentum_and_thin_the_cylinder(kernel):
    s = ib.Structure.from_bodies([ib.swimmer((40.0, 32.21))], ramp_steps=1000, kernel=kernel)
    sim = ib.ImmersedSim.create(s, 64, 96)
    out = sim.advance(1500)
    np.testing.assert_allclose(out.fluid_momentum(), sim.fluid_momentum(), atol=1e-10)
    assert not np.array_equal(out.X, replace(sim, structure=replace(s, kernel=4)).advance(1500).X)

    def drag_per_speed(width):
        ring = ib.cylinder((16.13, 16.29), 32 * np.sqrt(0.05 / np.pi), spacing=1.0)
        c = ib.Structure.from_bodies([ring], tether_k=0.05, kernel=width)
        run = ib.ImmersedSim.create(c, 32, 32, body_force=(2e-6, 0.0)).advance(5000)
        return -run.report["tether_fx"] / run.u.mean()

    # The 4-point kernel overpredicts drag by about 18 % here; narrower support, less.
    assert drag_per_speed(kernel) < 0.97 * drag_per_speed(4)
