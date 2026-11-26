from __future__ import annotations

import json
from dataclasses import replace

import numpy as np

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config
from hybrid_alife.experiments.runner import initialize_sim, run_experiment, step_sim
from hybrid_alife.metrics.energy_ledger import dissipated_heat, heat_metrics
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
