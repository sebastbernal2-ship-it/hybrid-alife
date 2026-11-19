from __future__ import annotations

from dataclasses import replace

import numpy as np

from hybrid_alife.experiments.runner import initialize_sim, load_config, step_sim
from hybrid_alife.physics import lbm
from hybrid_alife.physics.lbm import LiveLBM, agent_dipole_forces, agent_reaction_forces

N = 40  # periodic box side for field-pattern tests


def _box(h=N, w=N) -> LiveLBM:
    mask = np.zeros((w, h), np.uint8)
    f = np.zeros((w, h, 9))
    lbm._load().lbm_init(w, h, 1.0, np.zeros((w, h)), np.zeros((w, h)), f)
    z = np.zeros((w, h))
    return LiveLBM(f, mask, 0.1, 0.0, 0.05, z, z, np.ones((w, h)), np.zeros((h, w, 2)))


def _swimmers(n=12, seed=0):
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0.2, 0.8, (n, 2))
    swim = rng.uniform(-0.06, 0.06, (n, 2))
    return pos, swim, rng.uniform(0.6, 1.8, n), np.ones(n, bool)


def test_dipole_injects_zero_net_force_and_momentum_but_monopole_does_not():
    pos, swim, radii, alive = _swimmers()
    solid = np.zeros((N, N), np.uint8)
    dipole = agent_dipole_forces(pos, swim, radii, alive, solid, 2.0, 4.0, True)
    mono = agent_reaction_forces(pos, swim, radii, alive, solid, 2.0)
    assert np.abs(dipole.sum(axis=(0, 1))).max() < 1e-12
    assert np.abs(dipole).max() > 0 and np.abs(mono.sum(axis=(0, 1))).max() > 1e-3
    box = _box()
    p0 = box.momentum()
    np.testing.assert_allclose(box.advance(1, force=dipole).momentum() - p0, 0, atol=1e-10)
    assert np.abs(box.advance(1, force=mono).momentum() - p0).max() > 1e-3


def test_dipole_offsets_are_symmetric_integer_cells_around_the_agent():
    solid = np.zeros((N, N), np.uint8)
    one = (np.array([[0.51, 0.51]]), np.array([[0.05, 0.0]]), np.ones(1), np.ones(1, bool), solid)
    out = agent_dipole_forces(*one, 1.0, 4.0, True)  # agent in cell (20, 20): n = 2 along axis 0
    assert out[22, 20, 0] > 0 and out[18, 20, 0] < 0
    assert (np.abs(out).sum(-1) > 0).sum() == 2
    diag = (np.array([[0.51, 0.51]]), np.array([[0.04, 0.04]]), np.ones(1), np.ones(1, bool), solid)
    out = agent_dipole_forces(*diag, 1.0, 2.0, True)  # n = (1, 1): separation (2, 2)
    assert out[21, 21].tolist() > [0, 0] and out[19, 19].tolist() < [0, 0]


def test_even_gap_avoids_the_undamped_nyquist_mode_that_an_odd_gap_excites():
    n = 96
    z = np.zeros((n, n, 2))

    def nyquist_energy(gap: int) -> float:
        force = np.zeros((n, n, 2))
        force[47, 48] = (-0.3, 0.0)
        force[47 + gap, 48] = (0.3, 0.0)
        u = _box(n, n).advance(1, force=force).advance(320, force=z).velocity()
        k = np.fft.fftfreq(n) * n
        band = np.sqrt(k[:, None] ** 2 + k[None, :] ** 2) >= 40
        spectrum = np.abs(np.fft.fft2(u[..., 0])) ** 2 + np.abs(np.fft.fft2(u[..., 1])) ** 2
        return float(spectrum[band].sum() / n**2)

    assert nyquist_energy(1) > 50 * nyquist_energy(2)  # why the deposit quantizes to even gaps
    assert nyquist_energy(3) > 50 * nyquist_energy(4)


def test_dipole_skips_agents_on_solid_cells_keeping_zero_net_force():
    pos = np.array([[0.5, 0.5], [0.5, 0.5]])
    swim = np.array([[0.05, 0.0], [0.0, 0.05]])
    solid = np.zeros((N, N), np.uint8)
    solid[22, 20] = 1  # one end of agent 0's dipole (axis 0, length 4: cells 22 and 18)
    out = agent_dipole_forces(pos, swim, np.ones(2), np.ones(2, bool), solid, 1.0, 4.0, True)
    assert np.abs(out.sum(axis=(0, 1))).max() < 1e-12
    assert out[22, 20].tolist() == [0.0, 0.0] and out[18, 20].tolist() == [0.0, 0.0]  # skipped
    assert out[20, 22, 1] > 0 and out[20, 18, 1] < 0  # agent 1 (axis 1) still deposits
    same = agent_dipole_forces(
        pos[:1], swim[:1], np.ones(1), np.ones(1, bool), np.zeros((N, N), np.uint8), 1.0, 0.5, True
    )
    assert not same.any()  # both points on one cell cancel: nothing deposited
    edge = agent_dipole_forces(
        np.array([[0.01, 0.5]]), swim[:1], np.ones(1), np.ones(1, bool),
        np.zeros((N, N), np.uint8), 1.0, 4.0, False)
    assert not edge.any()  # point outside a non-toroidal domain: skipped


def _ring_speed(velocity: np.ndarray, radius: float) -> float:
    n = velocity.shape[0]
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    rows = np.round(n // 2 + radius * np.cos(theta)).astype(int) % n
    cols = np.round(n // 2 + radius * np.sin(theta)).astype(int) % n
    return float(np.linalg.norm(velocity[rows, cols], axis=-1).mean())


def _pulse(
    kind: str, steps: int, ell: float = 4.0, n: int = N, wait: int = 0, coeff: float = 5.0
) -> LiveLBM:
    pos = np.array([[0.5, 0.5]])
    swim = np.array([[0.06, 0.0]])  # swimming along +axis 0
    solid = np.zeros((n, n), np.uint8)
    one = (pos, swim, np.ones(1), np.ones(1, bool), solid)
    force = (
        agent_dipole_forces(*one, coeff, ell, True)
        if kind == "dipole"
        else agent_reaction_forces(*one, coeff)
    )
    fluid = _box(n, n).advance(steps, force=force)
    return fluid.advance(wait, force=np.zeros_like(force)) if wait else fluid


def test_dipole_conserves_momentum_and_far_field_decays_faster():
    # One-step pulse, then let momentum spread. Short times are dominated by acoustic
    # transients (the dipole is NOT smaller at 8 cells after ~10-40 steps), so compare after
    # the viscous spreading has set in (about 320 steps at nu = 0.1) in a box large enough
    # to avoid the periodic images.
    n, wait, c = 96, 320, 48
    mono = _pulse("monopole", 1, n=n, wait=wait)
    near = _pulse("dipole", 1, ell=2.0, n=n, wait=wait)
    far = _pulse("dipole", 1, ell=4.0, n=n, wait=wait)
    np.testing.assert_allclose(near.momentum(), 0, atol=1e-9)
    np.testing.assert_allclose(far.momentum(), 0, atol=1e-9)
    assert mono.momentum()[0] > 1e-3  # the monopole's momentum stays in the box
    r_mono = _ring_speed(mono.velocity(), 8)
    r_near, r_far = _ring_speed(near.velocity(), 8), _ring_speed(far.velocity(), 8)
    assert r_near < 0.4 * r_mono, (r_near, r_mono)
    assert r_near < r_far < 0.6 * r_mono  # larger dipole length, larger far field
    assert c == n // 2


def test_pusher_dipole_flow_pattern_outflow_on_axis_inflow_from_sides():
    v = _pulse("dipole", 15, coeff=2.0).velocity()  # kept below the low-Mach limit
    c, d = N // 2, 6
    assert v[c + d, c, 0] > 0 and v[c - d, c, 0] < 0  # ahead: pushed forward; behind: backward
    assert v[c, c + d, 1] < 0 and v[c, c - d, 1] > 0  # sides: flowing in toward the axis
    ahead = v[c + d, c, 0]
    assert ahead > 5 * abs(v[c + d, c, 1]) + 1e-12  # on-axis flow is along the axis


def _config(coupling: str):
    cfg = load_config("configs/ablation_lbm_live.yaml")
    world = replace(cfg.world, width=14, height=14, curvature_regime="default",
                    lbm_agent_drag=2.0, lbm_agent_coupling=coupling)
    return replace(cfg, world=world, embodied=replace(cfg.embodied, population_size=8),
                   avida=replace(cfg.avida, enabled=False))


def test_dipole_run_stays_finite_for_200_world_steps_with_agents():
    cfg = _config("dipole")
    state = initialize_sim(cfg)
    counter = 0
    for _ in range(200):
        state, counter = step_sim(state, cfg, counter)
        assert np.isfinite(np.asarray(state.world.flow)).all()
    assert np.isfinite(state.lbm.f).all()


def test_monopole_remains_the_default_and_matches_the_direct_function():
    assert load_config("configs/ablation_lbm_live.yaml").world.lbm_agent_coupling == "monopole"
    cfg = _config("monopole")
    state = initialize_sim(cfg)
    counter = 0
    for _ in range(3):
        state, counter = step_sim(state, cfg, counter)
    assert np.abs(state.lbm.pending_force.sum(axis=(0, 1))).max() > 0  # net force, as before
