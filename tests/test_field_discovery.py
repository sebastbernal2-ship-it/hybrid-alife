from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from hybrid_alife.experiments.field_discovery import (
    _condition_configs,
    _counterfactual_information_gain,
    _evaluation_flow,
    _probe_diagnostics,
    run_evaluation,
    summarize_results,
)
from hybrid_alife.experiments.runner import load_config


def test_no_field_and_decorrelated_conditions_change_only_target_access():
    cfg = load_config("configs/field_discovery_direct.yaml")
    no_field = _condition_configs(cfg, "no_field")
    decorrelated = _condition_configs(cfg, "decorrelated")

    assert no_field.embodied.flow_movement_scale == 0.0
    assert no_field.embodied.probe_signal_scale == 0.0
    assert "flow" in no_field.embodied.hidden_field_groups
    assert no_field.world.flow_advection_strength == 0.0
    assert no_field.world.resource_advection_strength == 0.0
    assert cfg.world.resource_advection_strength == 0.25
    assert no_field.world.resource_regen == cfg.world.resource_regen
    assert no_field.avida == cfg.avida
    assert decorrelated.embodied.shuffle_flow_observation is True
    assert decorrelated.embodied.flow_movement_scale == cfg.embodied.flow_movement_scale
    assert decorrelated.world == cfg.world


def test_counterfactual_information_gain_detects_spatial_flow_outcome_coupling():
    rng = np.random.default_rng(5)
    magnitude = rng.uniform(size=(16, 16))
    flow = np.stack((magnitude, np.zeros_like(magnitude)), axis=-1)
    gain = _counterfactual_information_gain(flow, magnitude)
    assert gain > 0.0


def test_probe_diagnostics_measure_signal_and_probe_cost():
    diagnostics = _probe_diagnostics(
        np.asarray([0.0, 1.0, 1.0]),
        np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]]),
        np.asarray([0.0, 1.0, 2.0]),
    )
    assert diagnostics["probe_rate"] == 2.0 / 3.0
    assert diagnostics["probe_signal_mean"] > 0.0
    assert diagnostics["probe_outcome_information"] >= 0.0


def test_intervention_flow_reverses_at_midpoint_and_transfer_scales():
    flow = np.array([[[1.0, -2.0]]])
    assert np.array_equal(
        _evaluation_flow(flow, step=1, split=2, reverse_flow=True, flow_scale=1.0), flow
    )
    assert np.array_equal(
        _evaluation_flow(flow, step=2, split=2, reverse_flow=True, flow_scale=1.0), -flow
    )
    assert np.array_equal(
        _evaluation_flow(flow, step=0, split=2, reverse_flow=False, flow_scale=1.5),
        flow * 1.5,
    )


def test_evaluation_uses_simulator_conditions_and_is_reproducible():
    first = run_evaluation(seed=7, steps=2, training_generations=1)
    second = run_evaluation(seed=7, steps=2, training_generations=1)

    assert first == second
    assert first["schema_version"] == 4
    assert first["training_generations"] == 1
    assert set(first["conditions"]) == {
        "direct", "indirect", "decorrelated", "no_field", "intervention"
    }
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
    assert first["conditions"]["intervention"]["config"]["flow_reversed"] is True
    assert first["conditions"]["decorrelated"]["config"]["shuffle_flow_observation"]
    assert first["conditions"]["no_field"]["config"]["flow_movement_scale"] == 0.0
    assert first["conditions"]["no_field"]["config"]["hidden_field_groups"] == [
        "flow", "shear", "shear_grad", "lift"
    ]
    assert "transfer" in first["conditions"]["direct"]
    assert "counterfactual_flow_information_gain" in first["conditions"]["direct"]
    assert "probe_diagnostics" in first["conditions"]["direct"]
    assert first["conditions"]["direct"]["probe_diagnostics"]["probe_rate"] >= 0.0
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
    assert set(result["conditions"]) == {
        "direct", "indirect", "decorrelated", "no_field", "intervention"
    }
    assert result["conditions"]["direct"]["config"]["file"] == "field_discovery_direct.yaml"
