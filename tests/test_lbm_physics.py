import numpy as np

from hybrid_alife.physics import lbm


def test_poiseuille_matches_analytic_profile():
    h, w, nu, g = 34, 8, 0.1, 1e-5
    u, v, _ = lbm.solve(lbm.channel_mask(h, w), nu, (g, 0.0), 6000)
    # Halfway bounce-back puts the walls midway between rows 0/1 and h-2/h-1.
    y = np.arange(1, h - 1) - 0.5
    height = h - 2
    analytic = g / (2 * nu) * y * (height - y)
    sim = u[1:-1, w // 2]
    assert np.max(np.abs(sim - analytic)) / analytic.max() < 0.02
    assert np.abs(v).max() < 1e-8


def test_mass_conserved_and_obstacle_blocks_flow():
    mask = lbm.channel_mask(24, 48, obstacle_radius=4)
    u, v, rho = lbm.solve(mask, 0.1, (1e-5, 0.0), 1500)
    fluid = mask == 0
    assert abs(rho[fluid].sum() - fluid.sum()) / fluid.sum() < 1e-6
    assert np.all(u[mask == 1] == 0)
    assert np.abs(v[fluid]).max() > 1e-6  # flow deflects around the obstacle


def test_rejects_unstable_viscosity():
    import pytest

    with pytest.raises(ValueError):
        lbm.solve(lbm.channel_mask(8, 8), 0.0, (0, 0), 1)


def test_world_uses_lbm_flow_when_configured():
    import jax

    from hybrid_alife.types import WorldConfig
    from hybrid_alife.world.env import initialize_world

    kw = dict(width=16, height=16, toroidal=False, resource_channels=1, hazard_channels=1,
              max_agents=2, max_digital_organisms=2, flow_noise_std=0.0,
              concentration_decay=0.9, concentration_diffusion=0.05)
    key = jax.random.PRNGKey(0)
    proxy = initialize_world(WorldConfig(**kw), key).flow
    lbm_flow = initialize_world(WorldConfig(**kw, flow_solver="lbm"), key).flow
    assert float(abs(lbm_flow).max()) == 1.0 and not np.allclose(proxy, lbm_flow)
    # no-slip: flow vanishes at the wall rows
    assert float(abs(lbm_flow[0]).max()) == 0.0


def test_rejects_supersonic_force_and_bad_mask_and_unknown_solver():
    import pytest

    from hybrid_alife.types import WorldConfig

    with pytest.raises(ValueError, match="low-Mach"):
        lbm.solve(lbm.channel_mask(16, 16), 0.1, (0.5, 0.0), 500)
    with pytest.raises(ValueError, match="2-D"):
        lbm.solve(np.zeros((4, 4, 2)), 0.1, (0.0, 0.0), 1)
    kw = dict(width=8, height=8, toroidal=False, resource_channels=1, hazard_channels=1,
              max_agents=1, max_digital_organisms=1, flow_noise_std=0.0,
              concentration_decay=0.9, concentration_diffusion=0.05)
    with pytest.raises(ValueError, match="flow_solver"):
        WorldConfig(**kw, flow_solver="lbn")


def test_failed_native_build_raises_clear_error(monkeypatch, tmp_path):
    import subprocess

    import pytest

    def boom(*args, **kwargs):
        raise subprocess.CalledProcessError(2, "make", stderr="g++: command not found")

    monkeypatch.setattr(lbm, "_lib", None)
    monkeypatch.setattr(lbm, "_LIB_PATH", tmp_path / "missing.so")
    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RuntimeError, match="g\\+\\+ and make"):
        lbm.solve(lbm.channel_mask(8, 8), 0.1, (0.0, 0.0), 1)
