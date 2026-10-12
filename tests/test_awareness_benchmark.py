from __future__ import annotations

import json

import pytest

from hybrid_alife.experiments.awareness_benchmark import (
    Action,
    BenchmarkConfig,
    WorldKind,
    evaluate_benchmark,
    run_episode,
    validate_environment_ladder,
)


def test_world_kinds_are_explicit_and_deterministic() -> None:
    first = run_episode(WorldKind.SOLVABLE, seed=7, policy="probe_then_exploit")
    second = run_episode(WorldKind.SOLVABLE, seed=7, policy="probe_then_exploit")
    assert first == second
    assert {kind.value for kind in WorldKind} == {
        "solvable",
        "surface_shift",
        "reversal",
        "deceptive",
        "impossible",
        "non_identifiable",
    }


def test_probe_reveals_solvable_rule_and_costs_reward() -> None:
    result = run_episode(WorldKind.SOLVABLE, seed=3, policy="probe_then_exploit")
    assert result["informative_probe_count"] == 1
    assert result["information_gain"] > 0.0
    assert result["probe_count"] == 1
    assert result["correct_exploit_rate"] == 1.0


def test_impossible_world_has_no_informative_signal() -> None:
    result = run_episode(WorldKind.IMPOSSIBLE, seed=3, policy="probe_then_exploit")
    assert result["informative_probe_count"] == 0
    assert result["information_gain"] == 0.0
    assert result["structure_rate"] == 0.0
    assert result["abstain_rate"] > 0.0


def test_non_identifiable_world_requires_targeted_probe() -> None:
    generic = run_episode(WorldKind.NON_IDENTIFIABLE, seed=3, policy="probe_then_exploit")
    targeted = run_episode(WorldKind.NON_IDENTIFIABLE, seed=3, policy="targeted_probe")
    assert generic["informative_probe_count"] == 0
    assert targeted["informative_probe_count"] >= 1
    assert targeted["information_gain"] > generic["information_gain"]


def test_reversal_changes_rule_and_reports_recovery() -> None:
    result = run_episode(
        WorldKind.REVERSAL,
        seed=3,
        policy="reversal_probe",
        config=BenchmarkConfig(horizon=8, reversal_step=4),
    )
    assert result["rule_before_reversal"] != result["rule_after_reversal"]
    assert result["reversal_recovery_steps"] is not None


def test_environment_ladder_validity_gate_passes() -> None:
    report = validate_environment_ladder(seeds=(0, 1, 2))
    assert report["valid"] is True
    assert report["checks"]["oracle_beats_no_signal"] is True
    assert report["checks"]["impossible_has_no_signal"] is True
    assert report["checks"]["surface_shift_preserves_rule"] is True
    assert report["checks"]["reversal_changes_rule"] is True
    assert report["checks"]["non_identifiable_requires_targeted_probe"] is True


def test_benchmark_artifact_is_json_serializable() -> None:
    artifact = evaluate_benchmark(seeds=(0, 1), horizon=6)
    encoded = json.dumps(artifact)
    assert artifact["schema_version"] == 1
    assert "validity_gate" in artifact
    assert "worlds" in artifact
    assert encoded


@pytest.mark.parametrize("action", list(Action))
def test_actions_have_stable_wire_values(action: Action) -> None:
    assert isinstance(action.value, str)
    assert action.value
