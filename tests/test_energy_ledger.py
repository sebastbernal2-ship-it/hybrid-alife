from __future__ import annotations

import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_alife.agents.embodied import (
    apply_embodied_actions,
    apply_reproduction,
    embodied_action_dim,
)
from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim, run_experiment, step_sim
from hybrid_alife.metrics.energy_ledger import (
    audit_energy_ledger,
    dissipated_heat,
    heat_metrics,
    new_ledger,
)
from hybrid_alife.replay.checkpoint import load_checkpoint, restore_sim_state, save_checkpoint


def _config(**embodied):
    base = embodied_law_config(
        seed=0, generations=1, population=6, steps_per_generation=3, size=10,
        proprioception=True)
    return replace(base, embodied=replace(base.embodied, **embodied))


def test_heat_equals_costs_when_nothing_else_moves_energy():
    cfg = _config(energy_ledger=True, basal_metabolic_cost=0.25, cost_eat=0.0,
                  cost_attack=0.0, cost_reproduce=0.0, cost_terraform=0.0, cost_emit=0.0,
                  cost_move_scale=0.0, flow_alignment_reward=0.0, displacement_reward=0.0)
    state = initialize_sim(cfg)
    state.world.hazards = state.world.hazards * 0.0
    cfg = replace(cfg, world=replace(cfg.world, hazard_decay=0.0))
    n_alive = int(state.embodied.alive.sum())
    state, _ = step_sim(state, cfg, 6, allow_reproduction=False)
    np.testing.assert_allclose(dissipated_heat(state.embodied.ledger), 0.25 * n_alive, rtol=1e-5)


def test_heat_excludes_what_a_dying_agent_could_not_pay():
    cfg = _config(energy_ledger=True, basal_metabolic_cost=1000.0)
    state = initialize_sim(cfg)
    held = float(state.embodied.energy.sum())
    state.world.resources = state.world.resources * 0.0
    cfg = replace(cfg, world=replace(cfg.world, resource_regen=0.0),
                  embodied=replace(cfg.embodied, flow_alignment_reward=0.0,
                                   displacement_reward=0.0))
    state, _ = step_sim(state, cfg, 6, allow_reproduction=False)
    assert int(state.embodied.alive.sum()) == 0
    np.testing.assert_allclose(dissipated_heat(state.embodied.ledger), held, rtol=1e-4)


def test_ledger_is_opt_in_and_does_not_change_dynamics():
    states = []
    for on in (False, True):
        cfg = _config(energy_ledger=on)
        state = initialize_sim(cfg)
        for _ in range(3):
            state, _ = step_sim(state, cfg, 6, allow_reproduction=False)
        states.append(state)
    off, on = states
    assert off.embodied.ledger is None and heat_metrics(off) == {}
    np.testing.assert_array_equal(off.embodied.energy, on.embodied.energy)
    assert heat_metrics(on)["embodied_dissipated_heat"] > 0.0


def test_runner_logs_heat_and_old_checkpoints_still_load(tmp_path):
    cfg = _config(energy_ledger=True)
    cfg = replace(cfg, output_dir=str(tmp_path),
                  logging=replace(cfg.logging, metrics_every=1, checkpoint_every_generations=1))
    state = run_experiment(cfg)
    rows = [json.loads(line) for line in
            (tmp_path / cfg.run_name / "metrics.jsonl").read_text().splitlines()]
    assert any(r.get("embodied_dissipated_heat", 0.0) > 0.0 for r in rows)
    path = save_checkpoint(tmp_path / "c.pkl", state, {})
    payload = load_checkpoint(path)
    restored = restore_sim_state(payload)
    assert dissipated_heat(restored.embodied.ledger) == dissipated_heat(state.embodied.ledger)
    del payload["embodied"]["ledger"]  # a checkpoint written before the field existed
    assert restore_sim_state(payload).embodied.ledger is None


def test_avida_heat_total_is_reported_when_the_energy_economy_is_on():
    from types import SimpleNamespace

    state = SimpleNamespace(embodied=None, avida=SimpleNamespace(heat=np.float32(2.5)))
    assert heat_metrics(state) == {"avida_dissipated_heat": 2.5}


def _everything_on():
    cfg = _config(
        energy_ledger=True, thermo_efficiency=0.6, flow_alignment_reward=0.4,
        displacement_reward=2.0, predation_rate=0.3, predation_radius=0.5,
        predation_efficiency=0.5, max_energy=8.0, initial_energy=8.0,
        reproduce_energy_threshold=6.0, reproduction_energy_cost=3.0,
        repro_gate_threshold=0.0, basal_metabolic_cost=0.4, initial_alive_fraction=0.5,
        flow_head=False, flow_head_eat_gate=0.0, linear_policy=True,
        unit_flow_features=False)
    # Closed matter cycle, so eating intake can be checked against the world arrays.
    return replace(cfg, world=replace(
        cfg.world, resource_regen=0.0, metabolite_decay=1.0, resource_advection_strength=0.0))


def _matter(world) -> float:
    return float(np.asarray(world.resources, np.float64).sum()
                 + np.asarray(world.metabolites, np.float64).sum())


def test_balance_with_every_term_through_step_world_and_reproduction():
    cfg = _everything_on()
    state = initialize_sim(cfg)
    state.world.hazards = state.world.hazards + 0.05
    held0 = float(state.embodied.energy.sum())
    matter0 = _matter(state.world)
    counter, births = 6, 0
    for _ in range(12):
        state, counter = step_sim(state, cfg, counter)
        births = state.embodied_births_this_gen
        audit = audit_energy_ledger(state.embodied)
        assert abs(audit["residual"]) < 2e-3, audit
        # Independent of the booked intake: energy plus 5 x world matter only changes by
        # what crossed the boundary (rewards and endowments in, heat out).
        boundary = audit["external_work"] + audit["injected"] - audit["heat"]
        change = (audit["held"] - held0) + 5.0 * (_matter(state.world) - matter0)
        assert abs(change - boundary) < 5e-3, (change, boundary)
    assert births > 0
    for term in ("intake", "external_work", "predation_loss", "cap_loss",
                 "reproduction_cost", "injected", "costs", "damage"):
        assert audit[term] != 0.0, term


def test_balance_survives_end_of_generation_selection_and_reseeding(tmp_path):
    cfg = _everything_on()
    cfg = replace(cfg, output_dir=str(tmp_path),
                  evolution=replace(cfg.evolution, generations=3))
    state = run_experiment(cfg)
    assert abs(audit_energy_ledger(state.embodied)["residual"]) < 2e-3


def _pair(**kwargs):
    cfg = _config(
        energy_ledger=True, basal_metabolic_cost=0.0, cost_eat=0.0, cost_attack=0.0,
        cost_reproduce=0.0, cost_terraform=0.0, cost_emit=0.0, cost_move_scale=0.0,
        flow_alignment_reward=0.0, displacement_reward=0.0, **kwargs)
    state = initialize_sim(cfg)
    pop = state.embodied
    pop.positions = jnp.full((6, 2), 0.5)
    pop.alive = jnp.arange(6) < 2
    pop.energy = jnp.array([10.0, 20.0, 0.0, 0.0, 0.0, 0.0])
    pop.ledger = new_ledger(pop.energy)
    state.world.hazards = jnp.zeros_like(state.world.hazards)
    state.world.resources = jnp.zeros_like(state.world.resources)
    actions = jnp.zeros((6, embodied_action_dim(cfg.embodied))).at[0, 3].set(1.0)
    pop, _, _ = apply_embodied_actions(
        pop, actions, state.world, cfg.embodied, cfg.world, jax.random.PRNGKey(0))
    return cfg, pop


def test_predation_efficiency_and_cap_loss_match_hand_computed_values():
    _, pop = _pair(predation_rate=0.5, predation_radius=0.2, predation_efficiency=0.25)
    np.testing.assert_allclose(pop.energy[:2], [12.5, 10.0])  # takes 10, keeps 2.5
    assert audit_energy_ledger(pop)["predation_loss"] == 7.5
    _, legacy = _pair(predation_rate=0.5, predation_radius=0.2)
    np.testing.assert_allclose(legacy.energy[:2], [20.0, 10.0])
    _, capped = _pair(max_energy=15.0)
    assert audit_energy_ledger(capped)["cap_loss"] == 5.0
    with pytest.raises(ValueError, match="predation_efficiency"):
        _config(predation_efficiency=1.5)


def test_ledger_mode_makes_a_slot_zero_parent_pay_for_its_child():
    cfg, pop = _pair(reproduce_energy_threshold=5.0, reproduction_energy_cost=4.0)
    pop.energy = jnp.array([10.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    pop.ledger = new_ledger(pop.energy)
    pop, births = apply_reproduction(pop, jnp.ones(6), cfg.embodied, 6, jax.random.PRNGKey(1))
    assert births == 1
    np.testing.assert_allclose(pop.energy[:3], [6.0, 1.0, cfg.embodied.initial_energy * 0.5])
    assert audit_energy_ledger(pop)["residual"] == 0.0
