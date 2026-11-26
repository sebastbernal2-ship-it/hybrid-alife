"""Loose speed guard for the native Avida step.

It compares native with the eager JAX step inside one process, so runner speed
cancels out. The native step measures about 20x faster; the 4x floor only trips
on a gross regression (an accidental -O0 build, per-organism Python work, or the
native flag silently falling back to JAX). If the JAX path itself gets much
faster, re-measure before lowering the floor.
"""

import time

import jax
from test_avida_native import _coupled_setup, _run_steps

MIN_SPEEDUP = 4.0


def _best_step_seconds(setup, native, repeats=5):
    def step():
        pop, _ = _run_steps(*setup, native=native, steps=1)
        jax.block_until_ready(pop.merit)

    step()  # warm up: library load and per-op JAX compiles
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        step()
        times.append(time.perf_counter() - start)
    return min(times)  # the minimum is the least disturbed by other load


def test_native_avida_step_stays_much_faster_than_jax():
    setup = _coupled_setup(0, 0.0, False)
    jax_s = _best_step_seconds(setup, native=False)
    native_s = _best_step_seconds(setup, native=True)
    assert jax_s / native_s >= MIN_SPEEDUP, f"jax {jax_s:.4f}s, native {native_s:.4f}s"
