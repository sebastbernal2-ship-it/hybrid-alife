from __future__ import annotations

from dataclasses import replace

import numpy as np

from hybrid_alife.experiments.causal_rollout import (
    prediction_update_proxy,
    probe_cost,
    reversal_adaptation_latency,
    run_frozen_causal_rollout,
)
from hybrid_alife.experiments.runner import initialize_sim, load_config, run_experiment, step_sim


def test_frozen_rollout_separates_visible_trace_from_evaluator_labels(tmp_path) -> None:
    config = load_config("configs/field_discovery_indirect.yaml")
    config = replace(
        config,
        output_dir=str(tmp_path),
        run_name="causal-source",
        avida=replace(config.avida, enabled=False),
        embodied=replace(config.embodied, population_size=2),
        evolution=replace(config.evolution, generations=1, steps_per_generation=1),
    )
    trained = run_experiment(config)
    artifact = run_frozen_causal_rollout(trained, config, steps=4, reverse_flow=True)
    assert artifact["config"]["avida_enabled"] is False
    assert len(artifact["agent_visible"]["observations"]) == 4
    assert len(artifact["evaluator_only"]["local_flow"]) == 4
    assert artifact["metrics"]["flow_reversal_adaptation_latency_proxy"] in (None, 0, 1, 2)


def test_step_sim_capture_hook_receives_agent_visible_observation() -> None:
    config = load_config("configs/field_discovery_indirect.yaml")
    config = replace(
        config,
        avida=replace(config.avida, enabled=False),
        embodied=replace(config.embodied, population_size=2),
    )
    state = initialize_sim(config)
    captured = []
    step_sim(state, config, 100, allow_reproduction=False, observation_capture=captured.append)
    assert len(captured) == 1
    assert captured[0].shape[0] == 2
    assert captured[0].shape[1] > 0


def test_prediction_update_proxy_measures_alignment_change_after_probe() -> None:
    flows = np.asarray(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    moves = np.asarray(
        [
            [[-1.0, 0.0]],
            [[0.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
        ]
    )
    result = prediction_update_proxy(
        moves,
        flows,
        np.asarray([[0.0], [1.0], [0.0], [0.0]]),
        np.ones((4, 1), dtype=bool),
        window=2,
    )
    assert result["probe_events"] == 1
    assert result["pre_probe_alignment"] == -1.0
    assert result["post_probe_alignment"] == 1.0
    assert result["prediction_update"] == 2.0


def test_prediction_update_proxy_reports_censored_no_probe() -> None:
    zeros = np.zeros((3, 2, 2), dtype=np.float32)
    result = prediction_update_proxy(
        zeros,
        zeros,
        np.zeros((3, 2)),
        np.ones((3, 2), dtype=bool),
    )
    assert result["probe_events"] == 0
    assert result["prediction_update"] is None
    assert result["censored"] is True


def test_probe_cost_is_exact_gate_cost() -> None:
    assert probe_cost(np.asarray([[0.0, 1.0], [0.5, 1.0]]), 0.2) == 0.5


def test_reversal_latency_requires_consecutive_post_reversal_alignment() -> None:
    flows = np.asarray(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    moves = np.asarray(
        [
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[1.0, 0.0]],
            [[-1.0, 0.0]],
            [[-1.0, 0.0]],
        ]
    )
    assert reversal_adaptation_latency(moves, flows, reversal_step=2, consecutive=2) == 1
    assert reversal_adaptation_latency(moves, flows, reversal_step=2, consecutive=3) is None
