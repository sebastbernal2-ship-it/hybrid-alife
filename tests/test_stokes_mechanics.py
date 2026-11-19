from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim

RADII = (0.6, 1.8)


def _run(stokes: bool, thrust):
    base = embodied_law_config(
        seed=0, generations=1, population=2, steps_per_generation=1, size=10,
        proprioception=True)
    emb = replace(
        base.embodied, body_morphology=True, stokes_mechanics=stokes, basal_metabolic_cost=0.0,
        flow_movement_scale=0.0, cost_move_scale=0.5, flow_alignment_reward=0.0,
        displacement_reward=0.0)
    cfg = replace(base, embodied=emb, world=replace(base.world, toroidal=True))
    state = initialize_sim(cfg)
    pop = state.embodied
    pop.positions = jnp.array([[0.25, 0.25], [0.75, 0.75]], dtype=jnp.float32)
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.energy = jnp.full((2,), 20.0)
    pop.genomes = {**pop.genomes, "body_radius": jnp.array(RADII, dtype=jnp.float32),
                   "body_aspect": jnp.ones((2,), dtype=jnp.float32)}
    state.world.hazards = jnp.zeros_like(state.world.hazards)
    actions = jnp.zeros((2, embodied_action_dim(emb)), dtype=jnp.float32)
    actions = actions.at[:, 0].set(jnp.array(thrust, dtype=jnp.float32))
    start = np.asarray(pop.positions).copy()
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, emb, cfg.world, jnp.array([0, 1]))
    return np.asarray(pop.positions) - start, 20.0 - np.asarray(pop.energy)


def test_equal_thrust_moves_big_bodies_slower_by_the_radius_ratio():
    dx, _ = _run(True, [0.5, 0.5])
    np.testing.assert_allclose(dx[1, 0] / dx[0, 0], RADII[0] / RADII[1], rtol=1e-4)
    legacy, _ = _run(False, [0.5, 0.5])
    np.testing.assert_allclose(legacy[0, 0], legacy[1, 0], rtol=1e-5)


def test_cost_is_dissipated_power_so_equal_speed_costs_radius_times_more():
    # Thrust scaled with radius gives both bodies the same swim speed.
    dx, cost = _run(True, [0.2, 0.2 * RADII[1] / RADII[0]])
    np.testing.assert_allclose(dx[0, 0], dx[1, 0], rtol=1e-4)
    np.testing.assert_allclose(cost[1] / cost[0], RADII[1] / RADII[0], rtol=1e-3)



def test_cost_excludes_advection_and_flow_estimate_stays_exact():
    base = embodied_law_config(
        seed=0, generations=1, population=2, steps_per_generation=1, size=10,
        proprioception=True)
    emb = replace(
        base.embodied, body_morphology=True, stokes_mechanics=True, basal_metabolic_cost=0.0,
        flow_movement_scale=0.05, cost_move_scale=0.5, flow_alignment_reward=0.0,
        displacement_reward=0.0)
    cfg = replace(base, embodied=emb, world=replace(base.world, toroidal=True))
    state = initialize_sim(cfg)
    pop = state.embodied
    pop.positions = jnp.array([[0.25, 0.25], [0.75, 0.75]], dtype=jnp.float32)
    pop.alive = jnp.ones((2,), dtype=bool)
    pop.energy = jnp.full((2,), 20.0)
    pop.genomes = {**pop.genomes, "body_radius": jnp.array(RADII, dtype=jnp.float32),
                   "body_aspect": jnp.ones((2,), dtype=jnp.float32)}
    state.world.hazards = jnp.zeros_like(state.world.hazards)
    state.world.flow = jnp.broadcast_to(jnp.array([0.3, -0.1]), state.world.flow.shape)
    actions = jnp.zeros((2, embodied_action_dim(emb)), dtype=jnp.float32).at[:, 0].set(0.5)
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, emb, cfg.world, jnp.array([0, 1]))
    # Swim speed is thrust / radius, so the cost is dissipated swimming power only.
    swim = 0.5 * 0.05 / np.array(RADII)
    expected = 0.5 * np.array(RADII) * swim**2 / 0.05
    spent = 20.0 - np.asarray(pop.energy)
    np.testing.assert_allclose(spent, expected, atol=5e-6)  # float32 resolution near 20
    # velocity - thrust is exactly the advecting flow, which the flow head relies on.
    np.testing.assert_allclose(
        np.asarray(pop.previous_velocity - pop.previous_thrust),
        np.broadcast_to(0.05 * np.array([0.3, -0.1]), (2, 2)), atol=1e-6)
