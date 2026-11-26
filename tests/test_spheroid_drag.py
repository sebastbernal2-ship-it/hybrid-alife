from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim
from hybrid_alife.physics.spheroid_drag import drag_factors


def _closed_form_float64(aspect):
    """The same published formulas in float64, away from the cancellation region."""
    if aspect > 1:
        e = np.sqrt(1 - aspect**-2)
        log = np.log((1 + e) / (1 - e))
        x = 8 / 3 * e**3 / (-2 * e + (1 + e * e) * log)
        y = 16 / 3 * e**3 / (2 * e + (3 * e * e - 1) * log)
        return x * aspect ** (2 / 3), y * aspect ** (2 / 3)
    e = np.sqrt(1 - aspect**2)
    x = 4 / 3 * e**3 / ((2 * e * e - 1) * np.arcsin(e) + e * np.sqrt(1 - e * e))
    y = 8 / 3 * e**3 / ((2 * e * e + 1) * np.arcsin(e) - e * np.sqrt(1 - e * e))
    return x * aspect ** (-1 / 3), y * aspect ** (-1 / 3)


@pytest.mark.parametrize("aspect", [0.5, 0.7, 0.96, 0.975, 0.98, 1.02, 1.027, 1.1, 1.4, 4.0])
def test_float32_factors_match_float64_across_the_series_switch(aspect):
    np.testing.assert_allclose(drag_factors(aspect), _closed_form_float64(aspect), rtol=2e-5)


def test_slender_body_and_thin_disk_limits():
    # Independent asymptotics. Slender body (Batchelor 1970, Cox 1970), semi-axes a >> b:
    # F_along = 4 pi mu a U / (ln(2a/b) - 1/2), F_across = 8 pi mu a U / (ln(2a/b) + 1/2).
    q = 100.0
    a = q ** (2 / 3)  # semi-major axis in units of the equal-volume radius
    along, across = drag_factors(q)
    np.testing.assert_allclose(along, a * (2 / 3) / (np.log(2 * q) - 0.5), rtol=1e-3)
    np.testing.assert_allclose(across, a * (4 / 3) / (np.log(2 * q) + 0.5), rtol=1e-3)
    # Thin disk of radius c: 16 mu c U broadside, 32/3 mu c U edgewise.
    q = 1e-3
    c = q ** (-1 / 3)
    along, across = drag_factors(q)
    np.testing.assert_allclose(along, c * 16 / (6 * np.pi), rtol=2e-3)
    np.testing.assert_allclose(across, c * (32 / 3) / (6 * np.pi), rtol=2e-3)


def _step(aspect, thrusts, aspect_drag=True, radius=1.3):
    base = embodied_law_config(
        seed=0, generations=1, population=2, steps_per_generation=1, size=10,
        proprioception=True)
    emb = replace(
        base.embodied, body_morphology=True, stokes_mechanics=True, aspect_drag=aspect_drag,
        basal_metabolic_cost=0.0, flow_movement_scale=0.0, cost_move_scale=0.5,
        cost_eat=0.0, cost_attack=0.0, cost_reproduce=0.0, cost_terraform=0.0, cost_emit=0.0,
        flow_alignment_reward=0.0, displacement_reward=0.0)
    cfg = replace(base, embodied=emb)
    state = initialize_sim(cfg)
    pop = state.embodied
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.energy = jnp.full((2,), 1.0)
    pop.genomes = {**pop.genomes, "body_radius": jnp.full((2,), radius),
                   "body_aspect": jnp.full((2,), aspect)}
    state.world.hazards = jnp.zeros_like(state.world.hazards)
    state.world.resources = jnp.zeros_like(state.world.resources)
    out = []
    for thrust in thrusts:
        actions = jnp.zeros((2, embodied_action_dim(emb)), dtype=jnp.float32)
        actions = actions.at[:, 0:2].set(jnp.array(thrust, dtype=jnp.float32))
        energy = np.asarray(pop.energy)
        pop, _, _ = apply_embodied_actions(
            pop, actions, state.world, emb, cfg.world, jnp.array([0, 1]))
        out.append((np.asarray(pop.previous_thrust[0]), float(energy[0] - pop.energy[0]),
                    np.asarray(pop.positions)))
    return out


THRUSTS = ([0.9, 0.2], [-0.3, 0.8], [0.0, 0.0], [0.4, -0.7])


def test_aspect_one_is_bit_identical_to_the_radius_law():
    for (v_on, c_on, p_on), (v_off, c_off, p_off) in zip(
        _step(1.0, THRUSTS), _step(1.0, THRUSTS, aspect_drag=False), strict=True
    ):
        np.testing.assert_array_equal(v_on, v_off)
        np.testing.assert_array_equal(p_on, p_off)
        assert c_on == c_off


def test_drag_is_anisotropic_and_the_axis_follows_thrust():
    aspect, radius = 1.4, 1.3
    along, across = (float(v) for v in drag_factors(aspect))
    assert across > along
    # Swim along x, idle (axis kept), turn to y (still broadside), then continue along y.
    steps = _step(aspect, ([1, 0], [0, 0], [0, 1], [0, 1]), radius=radius)
    speeds = [np.linalg.norm(s[0]) for s in steps]
    thrust = 0.05
    np.testing.assert_allclose(speeds[0], thrust / (radius * along), rtol=1e-5)
    np.testing.assert_allclose(speeds[2], thrust / (radius * across), rtol=1e-5)
    np.testing.assert_allclose(speeds[3], speeds[0], rtol=1e-5)
    # Oblique thrust is deflected towards the easy axis.
    v = _step(aspect, ([1, 0], [0.5, 0.5]), radius=radius)[1][0]
    np.testing.assert_allclose(v[0] / v[1], across / along, rtol=1e-5)


def test_cost_is_force_dot_velocity():
    steps = _step(0.7, ([1, 0], [0.5, 0.5]))
    velocity, cost, _ = steps[1]
    force = np.array([0.5, 0.5]) * 0.05
    # cost = 0.5 * F.v / 0.05 with thrust F = radius-1 sphere drag units.
    np.testing.assert_allclose(cost, 0.5 * float(force @ velocity) / 0.05, rtol=2e-4)


def test_aspect_drag_requires_stokes_bodies_and_newborns_have_no_axis():
    from hybrid_alife.agents.embodied import apply_reproduction

    base = embodied_law_config(seed=0, generations=1, population=2, steps_per_generation=1,
                               size=10)
    with pytest.raises(ValueError, match="aspect_drag"):
        replace(base.embodied, aspect_drag=True)
    emb = replace(base.embodied, body_morphology=True, stokes_mechanics=True,
                  aspect_drag=True, reproduce_energy_threshold=1.0,
                  reproduction_energy_cost=0.5)
    pop = initialize_sim(replace(base, embodied=emb)).embodied
    pop.alive = jnp.array([True, False])
    pop.energy = jnp.array([10.0, 0.0])
    pop.body_axis = jnp.ones((2, 2))
    pop, births = apply_reproduction(pop, jnp.ones(2), emb, 2, jnp.array([0, 1], jnp.uint32))
    assert births == 1
    np.testing.assert_array_equal(pop.body_axis, [[1.0, 1.0], [0.0, 0.0]])
