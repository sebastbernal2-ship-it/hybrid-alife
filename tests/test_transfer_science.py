from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.experiments.transfer import evaluate_fixed_policy
from hybrid_alife.metrics.comm_benchmark import BenchmarkSpec, enumerate_meanings, run_benchmark
from hybrid_alife.metrics.communication import (
    continuous_channel_capacity,
    continuous_comm_summary,
    continuous_topsim,
    uniform_continuous_channel,
)
from hybrid_alife.replay.checkpoint import load_checkpoint
from hybrid_alife.runtime import resolve_backend

REPO_ROOT = Path(__file__).resolve().parents[1]


def _tiny_config(tmp_path: Path, name: str):
    cfg = load_config(REPO_ROOT / "configs/base.yaml")
    return replace(
        cfg,
        output_dir=str(tmp_path),
        run_name=name,
        evolution=replace(cfg.evolution, generations=1, steps_per_generation=1),
    )


def test_fixed_policy_transfer_uses_checkpoint_genomes_without_evolution(tmp_path: Path):
    source = _tiny_config(tmp_path, "source")
    run_experiment(source)
    checkpoint = tmp_path / "source" / "checkpoint_final.pkl"
    before = load_checkpoint(checkpoint)
    target = replace(
        source,
        run_name="target",
        world=replace(source.world, flow_noise_std=0.0, concentration_diffusion=0.0),
        embodied=replace(
            source.embodied,
            genome_mutation_prob=1.0,
            repro_gate_threshold=0.0,
            reproduce_energy_threshold=0.0,
        ),
    )

    metrics = evaluate_fixed_policy(checkpoint, target, steps=2)
    assert np.isfinite(list(metrics.values())).all()
    assert "continuous_topsim" in metrics
    assert "continuous_channel_capacity" in metrics
    after = load_checkpoint(checkpoint)
    for name in before["embodied"]["genomes"]:
        np.testing.assert_array_equal(
            before["embodied"]["genomes"][name], after["embodied"]["genomes"][name]
        )


def test_fixed_policy_transfer_rejects_controller_shape_mismatch(tmp_path: Path):
    source = _tiny_config(tmp_path, "source")
    run_experiment(source)
    target = replace(
        source,
        embodied=replace(
            source.embodied, hidden_size=source.embodied.hidden_size + 1
        ),
    )
    with pytest.raises(ValueError, match="incompatible shapes"):
        evaluate_fixed_policy(tmp_path / "source" / "checkpoint_final.pkl", target, steps=0)


def test_continuous_metrics_and_uniform_control():
    meanings = enumerate_meanings(BenchmarkSpec(n_attr=2, attr_vocab=3, replicates=3))
    messages = np.concatenate(
        [np.eye(3, dtype=np.float64)[meanings[:, 0]], np.eye(3, dtype=np.float64)[meanings[:, 1]]],
        axis=1,
    )
    assert continuous_topsim(meanings, messages) > 0.9
    assert continuous_channel_capacity(messages, meanings[:, 0]) > 0.1
    summary = continuous_comm_summary(meanings, messages)
    assert set(summary) == {"continuous_topsim", "continuous_channel_capacity"}
    uniform = uniform_continuous_channel(messages)
    assert continuous_topsim(meanings, uniform) == 0.0
    assert continuous_channel_capacity(uniform, meanings[:, 0]) == 0.0

    result = run_benchmark(
        BenchmarkSpec(n_attr=2, attr_vocab=3, msg_vocab=8),
        protocols=["compositional"],
    )
    controls = result["compositional"].controls
    assert {"neutral", "uniform"} <= set(controls)
    assert controls["uniform"]["continuous_channel_capacity"] == 0.0


def test_ablation_seed_schedule_supports_reproducible_ten_seed_runs():
    path = REPO_ROOT / "scripts/run_ablation_matrix.py"
    spec = importlib.util.spec_from_file_location("run_ablation_matrix", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = Namespace(seed_list=None, seed_start=10, seeds=10)
    assert module._seed_schedule(args) == list(range(10, 20))
    args.seed_list = [4, 8, 15, 16, 23, 42, 43, 44, 45, 46]
    assert module._seed_schedule(args) == args.seed_list


def test_seeded_selection_is_stable_across_python_hash_seeds():
    code = """
from dataclasses import replace
import numpy as np
import jax.numpy as jnp
from hybrid_alife.evolution.selection import select_and_mutate_embodied
from hybrid_alife.experiments.runner import initialize_sim, load_config

cfg = load_config('configs/base.yaml')
cfg = replace(
    cfg, embodied=replace(cfg.embodied, genome_mutation_prob=1.0, genome_mutation_std=1.0)
)
state = initialize_sim(cfg)
state.embodied.alive = state.embodied.alive.at[0].set(False)
state.embodied = select_and_mutate_embodied(
    state.embodied,
    jnp.arange(cfg.embodied.population_size, dtype=jnp.float32),
    cfg.embodied,
    cfg.evolution,
    jnp.array([0, 123], dtype=jnp.uint32),
)
print(','.join(
    f'{float(np.asarray(value).sum()):.8f}' for value in state.embodied.genomes.values()
))
"""
    outputs = []
    for hash_seed in ('1', 'random'):
        env = os.environ | {'PYTHONHASHSEED': hash_seed, 'PYTHONPATH': str(REPO_ROOT / 'src')}
        result = subprocess.run(
            [sys.executable, '-c', code],
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        outputs.append(result.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1]


def test_gpu_request_falls_back_to_cpu(monkeypatch):
    import hybrid_alife.runtime as runtime

    def unavailable(_name):
        raise RuntimeError("no GPU backend")

    monkeypatch.setattr(runtime.jax, "devices", unavailable)
    assert resolve_backend("gpu") == "cpu"
