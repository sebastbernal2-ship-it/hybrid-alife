#!/usr/bin/env python
"""Measure one reproducible CPU run and record its output shape."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.world.env import diffuse_decay


def run_benchmark(cfg: Any, run_dir: str | Path) -> dict[str, Any]:
    """Run ``cfg`` on CPU and return a stable, JSON-serializable result."""
    run_dir = Path(run_dir)
    benchmark_cfg = replace(cfg, output_dir=str(run_dir.parent), run_name=run_dir.name)
    steps = benchmark_cfg.evolution.generations * benchmark_cfg.evolution.steps_per_generation
    started = time.perf_counter()
    state = run_experiment(benchmark_cfg)
    elapsed = time.perf_counter() - started

    output_shape: dict[str, list[int] | None] = {
        "world": list(np.asarray(state.world.resources).shape),
        "embodied": (
            list(np.asarray(state.embodied.positions).shape)
            if state.embodied is not None
            else None
        ),
        "avida": (
            list(np.asarray(state.avida.genomes).shape) if state.avida is not None else None
        ),
    }
    return {
        "schema_version": 1,
        "platform": "cpu",
        "python": platform.python_version(),
        "jax_platform": "cpu",
        "seed": benchmark_cfg.seed,
        "generations": benchmark_cfg.evolution.generations,
        "steps_per_generation": benchmark_cfg.evolution.steps_per_generation,
        "steps": steps,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / steps if steps else None,
        "output_shape": output_shape,
    }


def benchmark_jit_and_seed_batch(cfg: Any, seeds: int = 10, repeats: int = 8) -> dict[str, Any]:
    """Benchmark kernel JIT and seed batching without adopting either path.

    The full simulator still has Python-side dataclass and archive boundaries.
    This measures only a representative pure world kernel and deterministic
    random-key generation, so the result cannot be misreported as full-run
    speedup.
    """
    if seeds < 1 or repeats < 1:
        raise ValueError("seeds and repeats must be positive")
    x = jnp.ones(
        (cfg.world.height, cfg.world.width, cfg.world.resource_channels), dtype=jnp.float32
    )

    def kernel(value: jax.Array) -> jax.Array:
        return diffuse_decay(
            value,
            decay=cfg.world.concentration_decay,
            diffusion=cfg.world.concentration_diffusion,
            toroidal=cfg.world.toroidal,
        )

    compiled = jax.jit(kernel)
    compiled(x).block_until_ready()
    eager_start = time.perf_counter()
    for _ in range(repeats):
        kernel(x).block_until_ready()
    eager_seconds = time.perf_counter() - eager_start
    jit_start = time.perf_counter()
    for _ in range(repeats):
        compiled(x).block_until_ready()
    jit_seconds = time.perf_counter() - jit_start

    shape = (cfg.world.height, cfg.world.width, cfg.world.resource_channels)
    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), seeds)
    sample = jax.jit(lambda ks: jax.vmap(lambda k: jax.random.normal(k, shape))(ks))
    sample(keys).block_until_ready()
    sequential_start = time.perf_counter()
    for key in keys:
        jax.random.normal(key, shape).block_until_ready()
    sequential_seconds = time.perf_counter() - sequential_start
    batched_start = time.perf_counter()
    sample(keys).block_until_ready()
    batched_seconds = time.perf_counter() - batched_start
    return {
        "scope": "kernel_only",
        "adoption": "not_adopted_until_full_run_profile",
        "seeds": seeds,
        "repeats": repeats,
        "jit": {
            "eager_seconds": eager_seconds,
            "compiled_seconds": jit_seconds,
            "speedup": eager_seconds / jit_seconds if jit_seconds else None,
        },
        "seed_batch": {
            "sequential_seconds": sequential_seconds,
            "batched_seconds": batched_seconds,
            "speedup": sequential_seconds / batched_seconds if batched_seconds else None,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/scaling_tiny.yaml")
    parser.add_argument("--run-dir", default="outputs/benchmarks/cpu_run")
    parser.add_argument("--output", default="outputs/benchmarks/cpu_baseline.json")
    parser.add_argument(
        "--compare-paths",
        action="store_true",
        help="Also benchmark representative CPU JIT and seed-batched kernels.",
    )
    parser.add_argument("--benchmark-seeds", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    result = run_benchmark(cfg, args.run_dir)
    if args.compare_paths:
        result["optimization_benchmark"] = benchmark_jit_and_seed_batch(
            cfg, seeds=args.benchmark_seeds
        )
    result["config"] = args.config
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Benchmark written to {output}")


if __name__ == "__main__":
    main()
