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


def test_negative_eat_gate_cannot_create_matter_in_thermo_mode():
    cfg, state, pop, actions = _crowded_state(EFFICIENCY)
    actions = actions.at[:, 2].set(-1.0)
    before = _mass(state.world)
    pop, world, _ = apply_embodied_actions(
        pop, actions, state.world, cfg.embodied, cfg.world, jnp.array([0, 1]))
    assert abs(_mass(world) - before) < 1e-6
    assert float(pop.energy.sum()) == 6 * 20.0


def test_closed_system_conserves_matter_plus_energy_over_many_steps():
    from hybrid_alife.world.env import step_world

    cfg, state, pop, actions = _crowded_state(EFFICIENCY)
    world_cfg = replace(cfg.world, resource_regen=0.0, metabolite_decay=1.0,
                        hazard_decay=0.0, resource_advection_strength=0.0)
    cfg = replace(cfg, world=world_cfg)
    # Spread the agents over a few cells so crowding and free cells are both exercised.
    pop.positions = jnp.array(
        [[0.55, 0.55], [0.55, 0.55], [0.35, 0.35], [0.35, 0.35], [0.75, 0.15], [0.75, 0.15]],
        dtype=jnp.float32)
    state.embodied = pop
    total0 = _mass(state.world) + float(pop.energy.sum()) / 5.0
    for step in range(25):
        pop, world, _ = apply_embodied_actions(
            state.embodied, actions, state.world, cfg.embodied, world_cfg, jnp.array([step, 1]))
        state.embodied, state.world = pop, world
        state = step_world(state, world_cfg)
    total = _mass(state.world) + float(state.embodied.energy.sum()) / 5.0
    assert abs(total - total0) < 1e-3 * total0
    assert float(state.world.metabolites.sum()) > 0.0  # waste really was produced


def test_thermo_efficiency_is_validated_at_construction():
    import pytest

    cfg, *_ = _crowded_state(0.0)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="thermo_efficiency"):
            replace(cfg.embodied, thermo_efficiency=bad)
