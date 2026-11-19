"""ctypes bridge to the native Avida update kernel (native/avida.cpp)."""

from __future__ import annotations

import ctypes
import subprocess
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).with_name("libavida.so")
_NATIVE_DIR = Path(__file__).resolve().parents[3] / "native"
_lib: ctypes.CDLL | None = None


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not _LIB_PATH.exists():
            try:
                subprocess.run(
                    ["make", "-C", str(_NATIVE_DIR)], check=True, capture_output=True, text=True
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(
                    "building the native Avida kernel failed (needs g++ and make; "
                    f"run `make -C native`):\n{detail}"
                ) from exc
        lib = ctypes.CDLL(str(_LIB_PATH))
        nd = np.ctypeslib.ndpointer
        i32, f32, u8 = (nd(t, flags="C_CONTIGUOUS") for t in (np.int32, np.float32, np.uint8))
        lib.avida_run_update.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            i32, i32, i32, i32, i32, i32, i32, i32, f32, i32, i32, i32,
            u8, i32, i32, i32, f32, ctypes.c_double, ctypes.c_uint64, u8,
        ]
        lib.avida_run_update.restype = None
        lib.avida_n_ops.argtypes = []
        lib.avida_n_ops.restype = ctypes.c_int
        from hybrid_alife.agents.avida_vm import INSTRUCTION_COUNT

        if lib.avida_n_ops() != INSTRUCTION_COUNT:
            raise RuntimeError(
                f"native Avida kernel has {lib.avida_n_ops()} opcodes but the Python VM has "
                f"{INSTRUCTION_COUNT}; rebuild with `make -C native` (stale libavida.so?)"
            )
        _lib = lib
    return _lib


def require_concrete(*arrays) -> None:
    """The native kernel runs on host numpy buffers, so it cannot be traced."""
    import jax

    if any(isinstance(a, jax.core.Tracer) for a in arrays):
        raise RuntimeError(
            "AvidaConfig.native=True cannot run under jax.jit/vmap/grad (the native kernel works "
            "on concrete host arrays). Set native=False for jit-compiled code such as POET."
        )


def check_state(genomes, lengths, registers) -> None:
    """Reject states the C++ kernel would index out of bounds on."""
    if genomes.ndim != 2 or genomes.shape[1] < 1:
        raise ValueError(f"genomes must be [N, max_len>=1], got {genomes.shape}")
    n, max_len = genomes.shape
    if lengths.shape != (n,) or registers.ndim != 2 or registers.shape[0] != n:
        raise ValueError("genome_lengths/registers do not match the population size")
    if registers.shape[1] < 2:
        raise ValueError("the native Avida kernel needs at least 2 registers")
    if lengths.size and int(lengths.max()) > max_len:
        raise ValueError(
            f"genome_lengths exceed max_genome_length={max_len} (max {int(lengths.max())})"
        )


def run_update(pop, cfg, budget, env_a, env_b, rewards, seed: int):
    """Run one update's cycles natively. Mutates `pop` in place; returns the H_DIVIDE mask."""
    import jax.numpy as jnp

    require_concrete(pop.genomes, pop.merit, budget)

    def arr(x, dtype):
        return np.array(x, dtype=dtype, order="C")  # always a writable copy

    genomes, copied, regs = arr(pop.genomes, np.int32), arr(pop.copied, np.int32), arr(
        pop.registers, np.int32)
    ip, rh, wh, cl, tasks, ia, ib = (
        arr(getattr(pop, k), np.int32)
        for k in ("ip", "read_head", "write_head", "copied_length", "tasks_completed",
                  "last_input_a", "last_input_b"))
    merit = arr(pop.merit, np.float32)
    n, max_len = genomes.shape
    lengths = arr(pop.genome_lengths, np.int32)
    check_state(genomes, lengths, regs)
    div = np.zeros(n, np.uint8)
    _load().avida_run_update(
        n, max_len, regs.shape[1], genomes, lengths, regs, ip, rh, wh,
        copied, cl, merit, tasks, ia, ib, arr(pop.alive, np.uint8), arr(budget, np.int32),
        arr(env_a, np.int32), arr(env_b, np.int32), arr(rewards, np.float32),
        float(cfg.point_mutation_prob), seed & 0xFFFFFFFFFFFFFFFF, div,
    )
    pop.copied, pop.registers = jnp.asarray(copied), jnp.asarray(regs)
    pop.ip, pop.read_head, pop.write_head = map(jnp.asarray, (ip, rh, wh))
    pop.copied_length, pop.tasks_completed = jnp.asarray(cl), jnp.asarray(tasks)
    pop.last_input_a, pop.last_input_b, pop.merit = map(jnp.asarray, (ia, ib, merit))
    return jnp.asarray(div.astype(bool))
