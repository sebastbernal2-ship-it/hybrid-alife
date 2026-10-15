from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from hybrid_alife.experiments.awareness_benchmark import (
    Action,
    BenchmarkConfig,
    WorldKind,
    action_from_logits,
    encode_observation,
    evaluate_benchmark,
    init_learned_controller,
    run_episode,
    train_controller,
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


def test_controller_observation_encoder_preserves_missing_values() -> None:
    encoded = encode_observation(
        {"step": 2, "surface": 3, "cue": None, "probe_result": 1},
        horizon=8,
    )
    assert encoded.shape == (6,)
    assert encoded.tolist() == [0.25, 1.0, 0.0, 0.0, 1.0, 1.0]


def test_controller_action_mapping_uses_stable_action_order() -> None:
    logits = np.asarray([-1.0, -2.0, 4.0, 0.0, 1.0, 2.0])
    assert action_from_logits(logits) is Action.PROBE


def test_learned_controller_rejects_wrong_action_head_shape() -> None:
    params = init_learned_controller(seed=5, hidden=8)
    params = {**params, "w_act": params["w_act"][..., :5]}
    with pytest.raises(ValueError, match="action head"):
        run_episode(
            WorldKind.SOLVABLE,
            seed=3,
            policy="learned",
            controller_params=params,
        )


def test_learned_controller_rollout_is_deterministic_and_covers_ladder() -> None:
    params = init_learned_controller(seed=5, hidden=8)
    first = {
        kind.value: run_episode(
            kind,
            seed=3,
            policy="learned",
            controller_params=params,
        )
        for kind in WorldKind
    }
    second = {
        kind.value: run_episode(
            kind,
            seed=3,
            policy="learned",
            controller_params=params,
        )
        for kind in WorldKind
    }
    assert first == second
    assert set(first) == {kind.value for kind in WorldKind}
    assert all("probe_rate" in result for result in first.values())


def test_controller_training_is_seed_deterministic() -> None:
    first = train_controller(seed=9, training_seeds=(0, 1), generations=2, population_size=4)
    second = train_controller(seed=9, training_seeds=(0, 1), generations=2, population_size=4)
    assert first[1] == second[1]
    for name in first[0]:
        assert np.array_equal(np.asarray(first[0][name]), np.asarray(second[0][name]))


def test_cli_writes_learned_controller_artifact(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "learned.json"
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "JAX_PLATFORMS": "cpu"}
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_awareness_benchmark.py"),
            "--seeds",
            "3",
            "--training-seeds",
            "0",
            "--horizon",
            "6",
            "--train-controller",
            "--training-generations",
            "1",
            "--population-size",
            "2",
            "--output",
            str(output),
        ],
        check=True,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["learned_controller"]["status"] == "controller_calibration_only"
    assert set(result["learned_controller"]["evaluation"]) == {
        kind.value for kind in WorldKind
    }


@pytest.mark.parametrize("action", list(Action))
def test_actions_have_stable_wire_values(action: Action) -> None:
    assert isinstance(action.value, str)
    assert action.value
