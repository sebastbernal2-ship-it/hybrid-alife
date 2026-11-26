#!/usr/bin/env python
"""Time one Avida update: JAX vs native, eager and under jax.jit (pure_callback).

Run: PYTHONPATH=src JAX_PLATFORMS=cpu python benchmarks/bench_avida_native.py
"""

from __future__ import annotations

import dataclasses
import statistics
import time

import jax

from hybrid_alife import poet  # also registers the state dataclasses as pytrees
from hybrid_alife.agents import avida_vm

SIZES = (8, 64, 256, 1024, 4096)
# The POET Avida config (4 cycles, genome capacity 24) and a heavier VM like configs/base.yaml.
WORKLOADS = {
    "poet": {},
    "heavy": {"genome_length": 32, "max_genome_length": 128, "cycles_per_update": 8,
              "max_cycles_per_update": 32},
}


def time_call(fn, repeats: int) -> float:
    """Median seconds per call, after one warm-up (compile) call."""
    jax.block_until_ready(fn())
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        jax.block_until_ready(fn())
        samples.append(time.perf_counter() - started)
    return statistics.median(samples)


def bench_step(n: int, over: dict) -> dict[str, float]:
    """One update with the POET Avida config (plus `over`) scaled to `n` organisms."""
    base = dataclasses.replace(poet._poet_avida_config(), population_size=n, **over)
    poet_cfg = poet.POETConfig(avida_enabled=True)
    env = poet.initialize_environment(poet_cfg, jax.random.PRNGKey(0))
    world, _ = poet.initialize_environment_world(env, poet_cfg, jax.random.PRNGKey(1))
    pop = avida_vm.initialize_avida_population(base, jax.random.PRNGKey(2))
    positions = jax.random.uniform(jax.random.PRNGKey(3), (n, 2))
    key = jax.random.PRNGKey(4)
    out = {}
    for native in (False, True):
        cfg = dataclasses.replace(base, native=native)
        name = "native" if native else "jax"

        def run(fn, cfg=cfg):
            result = fn(dataclasses.replace(pop), dataclasses.replace(world), cfg, n, key,
                        positions)
            return result[0].merit, result[1] if len(result) == 3 else None

        out[f"jit_{name}"] = time_call(lambda run=run: run(poet._step_avida_fast), 200)
        out[f"eager_{name}"] = time_call(
            lambda run=run: run(avida_vm.step_avida_population), 30 if native else 5)
    return out


def bench_poet_rollout() -> dict[str, float]:
    """A whole POET rollout (16 steps, one Avida update per step) with the stock config."""
    base = poet._poet_avida_config()
    cfg = poet.POETConfig(avida_enabled=True)
    env = poet.initialize_environment(cfg, jax.random.PRNGKey(0))
    params = poet._init_policy(cfg, jax.random.PRNGKey(1))
    original, out = poet._poet_avida_config, {}
    try:
        for native in (False, True):
            poet._poet_avida_config = lambda native=native: dataclasses.replace(
                base, native=native)

            def rollout():
                batch, _, _ = poet._collect_rollout(
                    params, env, cfg, jax.random.PRNGKey(2), stochastic=True, horizon=None)
                return batch.rewards

            out["native" if native else "jax"] = time_call(rollout, 15)
    finally:
        poet._poet_avida_config = original
    return out


def main() -> None:
    print("ms per update (median)")
    columns = ("jit_jax", "jit_native", "eager_jax", "eager_native")
    for workload, over in WORKLOADS.items():
        print(f"[{workload}]\nn     " + " ".join(f"{c:>13}" for c in columns))
        for n in SIZES:
            row = bench_step(n, over)
            print(f"{n:<6}" + " ".join(f"{1e3 * row[c]:13.3f}" for c in columns), flush=True)
    rollout = bench_poet_rollout()
    print("POET rollout ms: " + ", ".join(f"{k}={1e3 * v:.1f}" for k, v in rollout.items()))


if __name__ == "__main__":
    main()
