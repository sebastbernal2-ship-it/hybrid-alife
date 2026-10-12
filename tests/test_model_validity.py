from __future__ import annotations

from dataclasses import replace

import jax
import numpy as np

from hybrid_alife.poet import (
    _collect_rollout,
    admission_calibration,
    environment_causal_validity,
    initialize_environment,
    initialize_poet_state,
    mutate_environment,
    run_poet,
    terrain_spatial_structure,
)
from tests.test_poet import tiny_config


def test_environment_genes_have_causal_validity_checks() -> None:
    cfg = tiny_config()
    env = initialize_environment(cfg, jax.random.PRNGKey(4))
    report = environment_causal_validity(env, cfg, seed=7)
    assert set(report) >= {
        "goal_x",
        "goal_y",
        "resource_regeneration",
        "hazard_rate",
        "flow_x",
        "flow_y",
        "terrain",
    }
    assert all(item["changed"] for item in report.values())
    assert all(item["mechanism_changed"] for item in report.values())


def test_terrain_mutation_changes_spatial_structure() -> None:
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=1)
    child = mutate_environment(state.environments[0], cfg, jax.random.PRNGKey(9))
    assert terrain_spatial_structure(child.terrain) != terrain_spatial_structure(
        state.environments[0].terrain
    )


def test_admission_calibration_reports_three_baselines() -> None:
    cfg = tiny_config()
    env = initialize_environment(cfg, jax.random.PRNGKey(2))
    report = admission_calibration(env, cfg, seed=3)
    assert set(report["baselines"]) == {"random", "fixed_policy", "hand_coded"}
    assert all(np.isfinite(value) for value in report["baselines"].values())
    assert "criterion" in report
    assert report["calibration_seeds"] == [3, 4, 5]
    assert set(report["repeated_baselines"]) == {"random", "fixed_policy", "hand_coded"}
    assert all(len(scores) == 3 for scores in report["repeated_baselines"].values())
    assert report["criterion_calibrated"] is False


def test_forager_interaction_and_reproduction_are_recorded() -> None:
    cfg = replace(tiny_config(), action_count=8)
    state = initialize_poet_state(cfg, seed=0)

    def action_selector(_obs, _env, step, _key):
        return 6 if step == 0 else 7

    _, _, diagnostics = _collect_rollout(
        state.policies[0],
        state.environments[0],
        cfg,
        jax.random.PRNGKey(5),
        stochastic=False,
        horizon=None,
        action_selector=action_selector,
    )
    assert diagnostics["interaction_events"] > 0
    assert diagnostics["reproduction_events"] > 0


def test_pilot_persists_ecology_and_bidirectional_coupling(tmp_path) -> None:
    cfg = replace(tiny_config(), avida_enabled=True, track="avida_enabled")
    result = run_poet(cfg, seed=0, generations=1, out_dir=tmp_path)
    diagnostics = result.to_jsonable()["ecology_diagnostics"]
    assert diagnostics["resource_gather_events"] > 0
    assert diagnostics["controller_to_avida_events"] > 0
    assert diagnostics["avida_feedback"] > 0
    assert (tmp_path / "summary.json").exists()


def test_avida_shared_field_coupling_changes_fixed_seed_rollout() -> None:
    cfg = replace(tiny_config(), action_count=8, avida_enabled=True, track="avida_enabled")
    state = initialize_poet_state(cfg, seed=2)
    env = state.environments[0]
    def actions(_obs, _env, step, _key):
        return 4 if step % 2 == 0 else 6
    coupled_batch, _, _ = _collect_rollout(
        state.policies[0],
        env,
        cfg,
        jax.random.PRNGKey(11),
        stochastic=False,
        horizon=8,
        action_selector=actions,
    )
    control_batch, _, _ = _collect_rollout(
        state.policies[0],
        env,
        replace(cfg, avida_enabled=False),
        jax.random.PRNGKey(11),
        stochastic=False,
        horizon=8,
        action_selector=actions,
    )
    assert not np.array_equal(
        np.asarray(coupled_batch.observations), np.asarray(control_batch.observations)
    )
