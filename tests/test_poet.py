from __future__ import annotations

import json
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
    evaluate_transfer_matrix,
    initialize_environment_world,
    initialize_poet_state,
    load_poet_checkpoint,
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
