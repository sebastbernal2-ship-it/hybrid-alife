from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.avida_vm import initialize_avida_population
from hybrid_alife.poet import (
    POETConfig,
    PPOConfig,
    _poet_avida_config,
    _step_avida_fast,
    collect_rollout,
    evaluate_policy,
    evaluate_transfer_matrix,
    initialize_environment_world,
    initialize_poet_state,
    load_poet_checkpoint,
    minimal_criterion_admits,
    mutate_environment,
    run_poet,
)


def tiny_config() -> POETConfig:
    return POETConfig(
        width=6,
        height=6,
        terrain_channels=1,
        population_size=2,
        policy_hidden=8,
        action_count=5,
        rollout_steps=6,
        ppo=PPOConfig(learning_rate=0.01, epochs=2),
        avida_enabled=False,
    )


def test_paired_fixture_is_deterministic_and_ppo_changes_only_policy():
    cfg = tiny_config()
    first = run_poet(cfg, seed=3, generations=2)
    second = run_poet(cfg, seed=3, generations=2)
    assert first.to_jsonable() == second.to_jsonable()
    assert first.policy_updates > 0
    assert not np.array_equal(first.initial_policy, first.final_policy)
    assert first.environment_mutations > 0


def test_minimal_criterion_uses_raw_reward_lower_bound():
    cfg = tiny_config()
    assert minimal_criterion_admits(cfg.minimal_criterion, cfg)
    assert not minimal_criterion_admits(cfg.minimal_criterion - 0.01, cfg)


def test_minimal_criterion_rejects_candidate_without_mutating_population():
    cfg = POETConfig(**{**tiny_config().__dict__, "minimal_criterion": 1_000_000.0})
    result = run_poet(cfg, seed=0, generations=1)
    assert result.environment_candidates == 1
    assert result.environment_rejections == 1
    assert result.environment_admissions == 0
    assert result.environment_mutations == 0


def test_environment_mutation_is_bounded_and_changes_dynamics():
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=0)
    child = mutate_environment(state.environments[0], cfg, jax.random.PRNGKey(9))
    assert np.all(child.params >= np.asarray(cfg.environment_param_low))
    assert np.all(child.params <= np.asarray(cfg.environment_param_high))
    assert not np.array_equal(child.terrain, state.environments[0].terrain)
    assert child.dynamics_signature(cfg) != state.environments[0].dynamics_signature(cfg)


def test_environment_genome_is_injected_into_existing_world_state():
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=2)
    world, world_cfg = initialize_environment_world(
        state.environments[0], cfg, jax.random.PRNGKey(10)
    )
    assert world.terrain.shape == (cfg.height, cfg.width, 4)
    assert world.resources.shape[:2] == (cfg.height, cfg.width)
    assert world_cfg.width == cfg.width
    assert float(world.terrain[..., 0].mean()) == float(state.environments[0].terrain.mean())


def test_avida_track_rollout_uses_shared_world_state():
    cfg = POETConfig(**{**tiny_config().__dict__, "avida_enabled": True, "track": "avida_enabled"})
    state = initialize_poet_state(cfg, seed=2)
    batch = collect_rollout(
        state.policies[0], state.environments[0], cfg, jax.random.PRNGKey(11)
    )
    isolated = collect_rollout(
        state.policies[0], state.environments[0], tiny_config(), jax.random.PRNGKey(11)
    )
    assert np.isfinite(np.asarray(batch.returns)).all()
    assert not np.array_equal(np.asarray(batch.observations), np.asarray(isolated.observations))


def test_avida_updates_shared_concentration_and_metabolites():
    cfg = POETConfig(**{**tiny_config().__dict__, "avida_enabled": True, "track": "avida_enabled"})
    state = initialize_poet_state(cfg, seed=4)
    world, _ = initialize_environment_world(
        state.environments[0], cfg, jax.random.PRNGKey(12)
    )
    avida_cfg = _poet_avida_config()
    population = initialize_avida_population(avida_cfg, jax.random.PRNGKey(13))
    positions = jnp.broadcast_to(
        jnp.asarray([0.5, 0.5], dtype=jnp.float32), (population.alive.shape[0], 2)
    )
    concentration_before = np.asarray(world.concentration)
    metabolites_before = np.asarray(world.metabolites)
    _, updated_world, _ = _step_avida_fast(
        population, world, avida_cfg, 8, jax.random.PRNGKey(14), positions
    )
    assert np.max(np.abs(np.asarray(updated_world.concentration) - concentration_before)) > 0.0
    assert np.max(np.abs(np.asarray(updated_world.metabolites) - metabolites_before)) > 0.0


def test_transfer_uses_frozen_policy_parameters():
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=1)
    before = np.asarray(state.policies[0].w1)
    matrix = evaluate_transfer_matrix(state.policies, state.environments, cfg, seed=4)
    np.testing.assert_array_equal(before, state.policies[0].w1)
    assert matrix.shape == (cfg.population_size, cfg.population_size)


def test_transfer_evaluation_reports_raw_reward_return():
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=1)
    batch = collect_rollout(
        state.policies[0],
        state.environments[0],
        cfg,
        jax.random.PRNGKey(4),
        stochastic=False,
        horizon=cfg.transfer_steps,
    )
    np.testing.assert_allclose(
        evaluate_policy(state.policies[0], state.environments[0], cfg, seed=4),
        np.sum(np.asarray(batch.rewards)),
    )


def test_transfer_evaluation_uses_configured_horizon():
    cfg = tiny_config()
    state = initialize_poet_state(cfg, seed=1)
    short = evaluate_policy(
        state.policies[0],
        state.environments[0],
        replace(cfg, transfer_steps=1),
        seed=4,
    )
    long = evaluate_policy(
        state.policies[0],
        state.environments[0],
        replace(cfg, transfer_steps=cfg.rollout_steps),
        seed=4,
    )
    assert short != long


def test_persistent_avida_track_advances_and_records_lifecycle(tmp_path: Path):
    cfg = POETConfig(
        **{
            **tiny_config().__dict__,
            "avida_enabled": True,
            "avida_lifecycle": "persistent",
            "track": "avida_persistent",
        }
    )
    result = run_poet(cfg, seed=0, generations=2, out_dir=tmp_path)
    payload = json.loads((tmp_path / "summary.json").read_text())
    config = json.loads((tmp_path / "config.json").read_text())
    assert result.avida_lifecycle == "persistent"
    assert payload["avida_lifecycle"] == "persistent"
    assert config["config"]["avida_lifecycle"] == "persistent"
    assert config["source_commit"]


def test_avida_comparator_horizon_is_recorded(tmp_path: Path):
    cfg = POETConfig(
        **{**tiny_config().__dict__, "avida_enabled": True, "track": "avida_enabled"}
    )
    run_poet(cfg, seed=0, generations=5, out_dir=tmp_path)
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["avida_comparator_steps"] == 4


def test_archive_records_lineage_and_stepping_stone_transfer(tmp_path: Path):
    cfg = POETConfig(**{**tiny_config().__dict__, "minimal_criterion": -100.0})
    result = run_poet(cfg, seed=0, generations=2, out_dir=tmp_path)
    assert len(result.environment_archive) > cfg.population_size
    assert any(entry.parent_id is not None for entry in result.environment_archive)
    assert result.archive_transfer_history
    archive_transfer = json.loads((tmp_path / "archive_transfer.json").read_text())
    assert archive_transfer["mode"] == "stepping_stone_archive_transfer"
    assert archive_transfer["matrix"]


def test_heldout_environments_are_separate_from_training_archive(tmp_path: Path):
    cfg = POETConfig(**{**tiny_config().__dict__, "heldout_environments": 2})
    result = run_poet(cfg, seed=4, generations=1, out_dir=tmp_path)
    assert len(result.heldout_environments) == 2
    assert result.heldout_transfer_history
    assert all(
        not np.array_equal(np.asarray(heldout.terrain), np.asarray(entry.environment.terrain))
        for heldout in result.heldout_environments
        for entry in result.environment_archive
    )
    heldout_transfer = json.loads((tmp_path / "heldout_transfer.json").read_text())
    assert heldout_transfer["excluded_from_training"] is True
    assert len(heldout_transfer["matrix"][0]) == 2


def test_replication_metadata_is_persisted_with_effective_seed(tmp_path: Path):
    cfg = tiny_config()
    run_poet(
        cfg,
        seed=3,
        generations=0,
        out_dir=tmp_path,
        replication_metadata={
            "replication_id": "rep-2",
            "seed_offset": 100,
            "operator": "tester",
            "machine_label": "machine-b",
            "cache_cleared": True,
        },
    )
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["effective_seed"] == 103
    assert payload["replication_metadata"]["machine_label"] == "machine-b"
    assert payload["replication_metadata"]["cache_cleared"] is True


def test_campaign_writes_manifest_analysis_and_figures(tmp_path: Path):
    cfg = tiny_config()
    result = run_poet(cfg, seed=0, generations=2, out_dir=tmp_path)
    assert (tmp_path / "metrics.jsonl").exists()
    assert (tmp_path / "transfer_matrix.json").exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "figures" / "poet_progress.png").exists()
    assert (tmp_path / "poet_checkpoint.pkl").exists()
    checkpoint = load_poet_checkpoint(tmp_path / "poet_checkpoint.pkl")
    assert checkpoint["version"] == 1
    assert checkpoint["state"].generation == 2
    payload = json.loads((tmp_path / "summary.json").read_text())
    assert payload["track"] == "isolated"
    assert payload["generations"] == 2
    assert result.generation == 2
    assert payload["environment_mutations"] > 0
    assert payload["replication_metadata"]["replication_id"]
