from __future__ import annotations

from hybrid_alife.experiments.causal_controller import (
    evaluate_causal_controller,
    run_causal_episode,
    train_causal_controller,
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
