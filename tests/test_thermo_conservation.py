from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.embodied import apply_embodied_actions, embodied_action_dim
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim

EFFICIENCY = 0.4


def _crowded_state(thermo: float):
    base = embodied_law_config(
        seed=0, generations=1, population=6, steps_per_generation=1, size=10,
        proprioception=True)
    emb = replace(
        base.embodied, thermo_efficiency=thermo, basal_metabolic_cost=0.0, cost_eat=0.0,
        cost_move_scale=0.0, initial_energy=20.0)
    cfg = replace(base, embodied=emb)
    state = initialize_sim(cfg)
    pop = state.embodied
    pop.positions = jnp.full((6, 2), 0.55, dtype=jnp.float32)  # everyone in one cell
    pop.alive = jnp.ones((6,), dtype=bool)
    pop.energy = jnp.full((6,), 20.0)
    state.world.resources = jnp.full_like(state.world.resources, 0.5)
    state.world.hazards = jnp.zeros_like(state.world.hazards)
    state.world.metabolites = jnp.zeros_like(state.world.metabolites)
    actions = jnp.zeros((6, embodied_action_dim(emb)), dtype=jnp.float32).at[:, 2].set(1.0)
    return cfg, state, pop, actions


def _mass(world):
    return float(world.resources.sum() + world.metabolites.sum())


def test_legacy_eating_creates_matter_when_agents_share_a_cell():
    cfg, state, pop, actions = _crowded_state(0.0)
    before = _mass(state.world)
    _, world, _ = apply_embodied_actions(
        pop, actions, state.world, cfg.embodied, cfg.world, jnp.array([0, 1]))
    # Documented defect: six agents each draw 20% of the cell, so >100% is paid out.
    removed = float(state.world.resources.sum() - world.resources.sum())
    gained = float(pop.energy.sum() - 6 * 20.0) / 5.0
    assert gained > removed + 1e-3 or _mass(world) != before


def test_thermo_mode_conserves_matter_and_accounts_for_energy():
    cfg, state, pop, actions = _crowded_state(EFFICIENCY)
    before_mass = _mass(state.world)
    before_energy = float(pop.energy.sum())
    removed_probe = float(state.world.resources.sum())
    pop, world, _ = apply_embodied_actions(
        pop, actions, state.world, cfg.embodied, cfg.world, jnp.array([0, 1]))
    removed = removed_probe - float(world.resources.sum())
    assert removed > 0
    gained = float(pop.energy.sum()) - before_energy
    # Matter + energy/5 (the fixed mass-energy exchange rate) is the conserved quantity:
    # eaten mass leaves as assimilated energy plus metabolite waste, nothing else.
    assert abs(_mass(world) + gained / 5.0 - before_mass) < 1e-4
    np.testing.assert_allclose(gained, removed * 5.0 * EFFICIENCY, rtol=1e-4)
    assert float(world.resources.min()) >= 0.0
