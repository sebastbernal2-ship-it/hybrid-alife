from __future__ import annotations

import pytest

from hybrid_alife.experiments.causal_controller import (
    evaluate_causal_controller,
    run_causal_episode,
    train_causal_controller,
    train_policy_gradient_controller,
)
from hybrid_alife.experiments.causal_task import BenchmarkConfig, WorldKind


def test_causal_controller_rollout_is_deterministic_and_leak_free() -> None:
    params, _ = train_causal_controller(
        seed=5,
        training_seeds=(0,),
        generations=1,
        population_size=2,
        config=BenchmarkConfig(horizon=6),
    )
    first = run_causal_episode(WorldKind.SOLVABLE, 3, params, BenchmarkConfig(horizon=6))
    second = run_causal_episode(WorldKind.SOLVABLE, 3, params, BenchmarkConfig(horizon=6))
    assert first == second
    assert "hidden_rule" not in first["agent_trace"][0]
    assert "hidden_rule" in first["evaluator_trace"][0]


def test_causal_controller_evaluation_has_all_worlds_and_baselines() -> None:
    params, _ = train_causal_controller(
        seed=7,
        training_seeds=(0,),
        generations=1,
        population_size=2,
        config=BenchmarkConfig(horizon=6),
    )
    result = evaluate_causal_controller(
        params,
        seeds=(0,),
        config=BenchmarkConfig(horizon=6),
    )
    assert set(result["learned"]) == {kind.value for kind in WorldKind}
    assert {"oracle", "random", "abstain", "cue"} <= set(result["baselines"])
    assert result["status"] == "pilot_only"


def test_inquiry_shaped_training_is_explicit_and_deterministic() -> None:
    kwargs = {
        "seed": 9,
        "training_seeds": (0,),
        "generations": 1,
        "population_size": 2,
        "config": BenchmarkConfig(horizon=6),
        "objective": "inquiry_shaped",
    }
    first, first_meta = train_causal_controller(**kwargs)
    second, second_meta = train_causal_controller(**kwargs)
    assert first_meta == second_meta
    assert first_meta["objective"] == "inquiry_shaped"
    for name in first:
        assert (first[name] == second[name]).all()


def test_value_of_information_uses_observed_post_probe_gain() -> None:
    params, metadata = train_causal_controller(
        seed=11,
        training_seeds=(0,),
        generations=1,
        population_size=2,
        objective="value_of_information",
        config=BenchmarkConfig(horizon=6),
    )
    result = run_causal_episode(WorldKind.SOLVABLE, 0, params, BenchmarkConfig(horizon=6))
    assert metadata["objective"] == "value_of_information"
    assert "post_probe_return_gain" in result
    assert "hidden_rule" not in result["agent_trace"][0]


def test_policy_gradient_training_is_seed_deterministic() -> None:
    kwargs = {
        "seed": 13,
        "training_seeds": (0,),
        "updates": 1,
        "episodes_per_update": 2,
        "config": BenchmarkConfig(horizon=6),
    }
    first, first_meta = train_policy_gradient_controller(**kwargs)
    second, second_meta = train_policy_gradient_controller(**kwargs)
    assert first_meta == second_meta
    assert first_meta["trainer"] == "policy_gradient"
    assert first_meta["baseline"] == "batch_timestep_mean"
    for name in first:
        assert (first[name] == second[name]).all()


def test_unknown_training_objective_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown training objective"):
        train_causal_controller(
            seed=1,
            generations=1,
            population_size=2,
            objective="oracle_labels",
        )
