"""Correctness guardrails for control runs, replay, and measured execution."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from hybrid_alife.evolution import selection
from hybrid_alife.experiments.runner import (
    initialize_sim,
    load_config,
    run_experiment,
    step_sim,
    validate_runtime_invariants,
)
from hybrid_alife.experiments.shadow import paired_activity_from_lineage_logs, run_shadow
from hybrid_alife.replay.checkpoint import (
    load_checkpoint,
    restore_sim_state,
    save_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(tmp_path: Path, run_name: str = "solidness"):
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    return replace(
        cfg,
        output_dir=str(tmp_path),
        run_name=run_name,
        evolution=replace(cfg.evolution, generations=1, steps_per_generation=1),
    )


def test_shadow_run_writes_a_paired_control(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    original_select = selection.tournament_select
    state = run_shadow(cfg)
    run_dir = tmp_path / "solidness_shadow"
    paired = paired_activity_from_lineage_logs(
        [{"lineage_ids": np.array([1, 1]), "alive": np.array([True, False])}],
        [{"lineage_ids": np.array([1, 1]), "alive": np.array([True, False])}],
    )

    assert selection.tournament_select is original_select
    assert len(paired) == 1
    assert state.generation == 0
    assert (run_dir / "metrics.jsonl").exists()
    assert (run_dir / "checkpoint_final.pkl").exists()
    last = json.loads((run_dir / "metrics.jsonl").read_text().splitlines()[-1])
    assert last["kind"] == "generation"


def test_runtime_invariants_fail_fast_on_nonfinite_state(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    state = initialize_sim(cfg)
    validate_runtime_invariants(state, cfg)

    state.world.resources = state.world.resources.at[0, 0, 0].set(jnp.nan)
    with pytest.raises(RuntimeError, match="world.resources contains non-finite values"):
        validate_runtime_invariants(state, cfg)


def test_checkpoint_replay_continuation_is_deterministic(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path)
    state = initialize_sim(cfg)
    state, lineage_counter = step_sim(state, cfg, 100)
    checkpoint = save_checkpoint(tmp_path / "replay.pkl", state, {"seed": cfg.seed})
    replayed = restore_sim_state(load_checkpoint(checkpoint))

    expected, expected_counter = step_sim(state, cfg, lineage_counter)
    actual, actual_counter = step_sim(replayed, cfg, lineage_counter)

    assert expected_counter == actual_counter
    assert expected.step == actual.step
    assert expected.world.time == actual.world.time
    np.testing.assert_array_equal(np.asarray(expected.rng), np.asarray(actual.rng))
    np.testing.assert_array_equal(
        np.asarray(expected.world.concentration), np.asarray(actual.world.concentration)
    )
    np.testing.assert_array_equal(
        np.asarray(expected.embodied.positions), np.asarray(actual.embodied.positions)
    )
    np.testing.assert_array_equal(
        np.asarray(expected.avida.genomes), np.asarray(actual.avida.genomes)
    )


def test_real_run_produces_nonempty_visualization_artifacts(tmp_path: Path) -> None:
    cfg = _tiny_config(tmp_path, "visualization")
    run_experiment(cfg)
    run_dir = tmp_path / "visualization"
    subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "generate_report.py"), str(run_dir)],
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )

    for name in ("world_fields", "agents", "metrics", "map_elites"):
        artifact = run_dir / "plots" / f"{name}.png"
        assert artifact.is_file() and artifact.stat().st_size > 0
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "plots/world_fields.png" in report


def test_cpu_benchmark_records_runtime_and_output_shape(tmp_path: Path) -> None:
    script_path = REPO_ROOT / "scripts" / "benchmark_cpu.py"
    spec = importlib.util.spec_from_file_location("benchmark_cpu", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    cfg = _tiny_config(tmp_path, "benchmark")
    result = module.run_benchmark(cfg, tmp_path / "benchmark")

    assert result["platform"] == "cpu"
    assert result["elapsed_seconds"] > 0.0
    assert result["steps"] == 1
    assert result["output_shape"]["world"] == [
        cfg.world.height,
        cfg.world.width,
        cfg.world.resource_channels,
    ]
    assert result["output_shape"]["embodied"] == [cfg.embodied.population_size, 2]
    assert result["output_shape"]["avida"] == [
        cfg.avida.population_size,
        cfg.avida.max_genome_length,
    ]

    comparison = module.benchmark_jit_and_seed_batch(cfg, seeds=10, repeats=1)
    assert comparison["scope"] == "kernel_only"
    assert comparison["adoption"] == "not_adopted_until_full_run_profile"
    assert comparison["jit"]["speedup"] > 0.0
    assert comparison["seed_batch"]["speedup"] > 0.0
