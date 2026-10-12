from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from hybrid_alife.experiments.field_discovery import run_evaluation, summarize_results


def test_evaluation_uses_simulator_conditions_and_is_reproducible():
    first = run_evaluation(seed=7, steps=2, training_generations=1)
    second = run_evaluation(seed=7, steps=2, training_generations=1)

    assert first == second
    assert first["schema_version"] == 3
    assert first["training_generations"] == 1
    assert set(first["conditions"]) == {"direct", "indirect", "shuffled", "intervention"}
    for condition in first["conditions"].values():
        assert set(condition["final_metrics"]) == {
            "survival",
            "energy",
            "resource",
            "reproduction",
        }
        assert "score" in condition["pre_intervention"]
        assert "score" in condition["post_intervention"]
    assert first["conditions"]["intervention"]["config"]["intervention_step"] == 1
    assert first["conditions"]["shuffled"]["config"]["hidden_field_groups"]
    assert first["conditions"]["shuffled"]["config"]["shuffle_sixth_sense"] is True
    summary = summarize_results([first, first])
    assert summary["run_count"] == 2
    assert summary["conditions"]["direct"]["survival"]["n"] == 2
    assert summary["conditions"]["intervention"]["post_intervention_score"]["n"] == 2


def test_cli_writes_compact_pilot_artifact(tmp_path):
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "pilot.json"
    env = {**os.environ, "PYTHONPATH": str(root / "src"), "JAX_PLATFORMS": "cpu"}
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_field_discovery.py"),
            "--seed",
            "3",
            "--steps",
            "2",
            "--training-generations",
            "1",
            "--config-dir",
            str(root / "configs"),
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
    assert result["seed"] == 3
    assert set(result["conditions"]) == {"direct", "indirect", "shuffled", "intervention"}
    assert result["conditions"]["direct"]["config"]["file"] == "field_discovery_direct.yaml"
