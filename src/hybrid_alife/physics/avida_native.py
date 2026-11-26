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
            ctypes.c_void_p, ctypes.c_float, ctypes.c_float, ctypes.c_void_p,
        ]
        lib.avida_run_update.restype = None
        dd = ctypes.c_double
        lib.avida_divide.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            dd, dd, dd, ctypes.c_uint64, ctypes.c_int32, u8,
            i32, i32, i32, i32, i32, i32, i32, i32, i32, f32, i32, u8, i32, i32, i32, i32,
            ctypes.c_void_p,
        ]
        lib.avida_divide.restype = None
        lib.avida_couple.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            i32, i32, u8, ctypes.c_float, ctypes.c_float, f32, f32, f32,
        ]
        lib.avida_couple.restype = None
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


def is_traced(*trees) -> bool:
    """True if any array leaf is a JAX tracer (we are inside jit/vmap/grad)."""
    import jax

    return any(isinstance(x, jax.core.Tracer) for x in jax.tree_util.tree_leaves(trees))


def require_concrete(*arrays) -> None:
    """The low-level entry points below run on host buffers, so they cannot be traced."""
    if is_traced(*arrays):
        raise RuntimeError(
            "this native entry point cannot run under jax.jit/vmap/grad; under a transform use "
            "step_avida_population, which calls the kernel through jax.pure_callback"
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


def _call_run_update(lib, cfg, x, budget, env_a, env_b, rewards, seed, alive, div, energy,
                     totals) -> None:
    n, max_len = x["genomes"].shape
    lib.avida_run_update(
        n, max_len, x["registers"].shape[1], x["genomes"], x["genome_lengths"], x["registers"],
        x["ip"], x["read_head"], x["write_head"], x["copied"], x["copied_length"], x["merit"],
        x["tasks_completed"], x["last_input_a"], x["last_input_b"], alive, budget, env_a, env_b,
        rewards, float(cfg.point_mutation_prob), seed & 0xFFFFFFFFFFFFFFFF, div,
        energy.ctypes.data if energy is not None else None, float(cfg.instruction_energy_cost),
        float(cfg.energy_efficiency), totals.ctypes.data,
    )


def run_update(pop, cfg, budget, env_a, env_b, rewards, seed: int):
    """Eager-only: run just the cycle phase. Mutates `pop`; returns the H_DIVIDE mask."""
    import jax.numpy as jnp

    require_concrete(pop.genomes, pop.merit, budget)
    names = ("genomes", "genome_lengths", "registers", "ip", "read_head", "write_head",
             "copied", "copied_length", "tasks_completed", "last_input_a", "last_input_b")
    x = {k: np.array(getattr(pop, k), dtype=np.int32, order="C") for k in names}
    x["merit"] = np.array(pop.merit, dtype=np.float32, order="C")
    check_state(x["genomes"], x["genome_lengths"], x["registers"])
    energy_on = pop.energy is not None and cfg.instruction_energy_cost > 0.0
    energy = np.array(pop.energy, dtype=np.float32, order="C") if energy_on else None
    div = np.zeros(x["genomes"].shape[0], np.uint8)
    totals = np.zeros(2, np.float64)
    _call_run_update(
        _load(), cfg, x, np.array(budget, np.int32), np.array(env_a, np.int32),
        np.array(env_b, np.int32), np.array(rewards, np.float32), seed,
        np.array(pop.alive, np.uint8), div, energy, totals)
    if energy_on:
        pop.energy = jnp.asarray(energy)
        pop.energy_in = pop.energy_in + jnp.float32(totals[0])
        pop.heat = pop.heat + jnp.float32(totals[1])
    for name, value in x.items():
        if name not in ("genomes", "genome_lengths"):
            setattr(pop, name, jnp.asarray(value))
    return jnp.asarray(div.astype(bool))


def _call_divide(lib, cfg, x, next_lineage_start, seed, alive, div, energy) -> None:
    n, max_len = x["genomes"].shape
    lib.avida_divide(
        n, max_len, x["registers"].shape[1], x["memory"].shape[1], int(cfg.min_genome_length),
        max(1, cfg.genome_length // 8), float(cfg.insertion_prob), float(cfg.deletion_prob),
        float(cfg.duplication_prob), seed & 0xFFFFFFFFFFFFFFFF, next_lineage_start, div,
        x["genomes"], x["genome_lengths"], x["registers"], x["memory"], x["ip"], x["read_head"],
        x["write_head"], x["copied"], x["copied_length"], x["merit"], x["age"], alive,
        x["tasks_completed"], x["lineage_id"], x["parent_id"], x["lineage_depth"],
        energy.ctypes.data if energy is not None else None,
    )


def _call_couple(lib, cfg, gr, gc, alive, metab, conc, merit) -> None:
    h, w = metab.shape[0], metab.shape[1]
    gr, gc = np.ascontiguousarray(gr, np.int32), np.ascontiguousarray(gc, np.int32)
    if gr.shape != alive.shape or gc.shape != alive.shape:
        raise ValueError("positions must have one row per organism")
    if gr.size and not (0 <= gr.min() and gr.max() < h and 0 <= gc.min() and gc.max() < w):
        raise ValueError("organism grid cells fall outside the world")
    lib.avida_couple(
        alive.shape[0], h, w, metab.shape[-1], conc.shape[-1], gr, gc, alive,
        float(cfg.metabolite_uptake), float(cfg.metabolite_deposit), metab, conc, merit,
    )


# ---------------------------------------------------------------------------
# Whole update (eager and jax.jit)
# ---------------------------------------------------------------------------

_INT_FIELDS = ("genomes", "genome_lengths", "registers", "memory", "ip", "read_head",
               "write_head", "copied", "copied_length", "age", "lineage_id", "parent_id",
               "lineage_depth", "tasks_completed", "last_input_a", "last_input_b")
_ENERGY_FIELDS = ("energy", "energy_in", "heat")


def _host_update(cfg, inputs: dict) -> dict:
    """Run one full update on host arrays. Pure: copies every buffer, returns fresh ones.

    NumPy and C++ only: calling JAX from inside a host callback can deadlock.
    """
    x = {k: np.array(v, order="C") for k, v in inputs.items()}  # private writable copies
    lib = _load()
    genomes, regs, mem = x["genomes"], x["registers"], x["memory"]
    check_state(genomes, x["genome_lengths"], regs)
    n, max_len = genomes.shape
    if mem.ndim != 2 or mem.shape[0] != n:
        raise ValueError("memory must be [N, memory_size] matching the population")
    cl = x["copied_length"]
    if cl.size and (int(cl.min()) < 0 or int(cl.max()) > max_len):
        raise ValueError("copied_length must lie in [0, max_genome_length]")
    alive = x["alive"].astype(np.uint8)
    div = np.zeros(n, np.uint8)
    has_energy = "energy" in x
    paying = has_energy and cfg.instruction_energy_cost > 0.0
    totals = np.zeros(2, np.float64)
    _call_run_update(lib, cfg, x, x["budget"], x["env_a"], x["env_b"], x["rewards"],
                     int(x["seed"]), alive, div, x["energy"] if paying else None, totals)
    if paying:
        x["energy_in"] = np.asarray(x["energy_in"] + np.float32(totals[0]), np.float32)
        x["heat"] = np.asarray(x["heat"] + np.float32(totals[1]), np.float32)
    _call_divide(lib, cfg, x, int(x["next_lineage_start"]), int(x["seed_div"]), alive, div,
                 x["energy"] if has_energy else None)
    alive = alive.astype(bool)
    if cfg.instruction_energy_cost > 0.0:
        alive &= x["energy"] > 0.0  # out of energy: die at update end
    x["alive"] = alive
    x["age"] += alive.astype(np.int32)
    if "gr" in x:
        _call_couple(lib, cfg, x["gr"], x["gc"], alive.astype(np.uint8), x["metabolites"],
                     x["concentration"], x["merit"])
    return x


def step(pop, world, cfg, budget, env_a, env_b, rewards, seed, seed_div,
         next_lineage_start, positions) -> None:
    """One native update (cycles, division, death, aging, coupling). Mutates `pop`/`world`.

    Eager calls go straight to the host function; under jax.jit it is wrapped in
    ``jax.pure_callback`` (one callback per update, ``vmap`` runs it sequentially).
    All RNG state enters as the explicit ``seed``/``seed_div`` inputs, so the callback is pure
    and a jitted step reproduces the eager step.
    """
    import jax
    import jax.numpy as jnp

    from hybrid_alife.world.env import positions_to_grid

    n = pop.genomes.shape[0]
    inputs = {f: jnp.asarray(getattr(pop, f), jnp.int32) for f in _INT_FIELDS}
    inputs["merit"] = jnp.asarray(pop.merit, jnp.float32)
    inputs["alive"] = jnp.asarray(pop.alive, jnp.bool_)
    if pop.energy is not None:
        for f in _ENERGY_FIELDS:
            inputs[f] = jnp.asarray(getattr(pop, f), jnp.float32)
    outputs = list(inputs)
    inputs.update(
        budget=jnp.asarray(budget, jnp.int32), env_a=jnp.asarray(env_a, jnp.int32),
        env_b=jnp.asarray(env_b, jnp.int32), rewards=jnp.asarray(rewards, jnp.float32),
        seed=jnp.asarray(seed, jnp.int32), seed_div=jnp.asarray(seed_div, jnp.int32),
        next_lineage_start=jnp.asarray(next_lineage_start, jnp.int32),
    )
    coupled = positions is not None and positions.shape[0] == n
    if coupled:
        metab, conc = world.metabolites, world.concentration
        if metab.ndim != 3 or conc.ndim != 3 or conc.shape[:2] != metab.shape[:2]:
            raise ValueError("metabolites and concentration must be [H, W, C] on the same grid")
        # positions_to_grid clips to the grid, so these indices are always in bounds.
        grid = positions_to_grid(positions, *metab.shape[:2])
        inputs.update(gr=grid[:, 0], gc=grid[:, 1],
                      metabolites=jnp.asarray(metab, jnp.float32),
                      concentration=jnp.asarray(conc, jnp.float32))
        outputs += ["metabolites", "concentration"]

    def host(args):
        result = _host_update(cfg, args)
        return {k: result[k] for k in outputs}

    if is_traced(inputs):
        _load()  # build/load now so a failure raises while tracing, not inside the callback
        shapes = {k: jax.ShapeDtypeStruct(inputs[k].shape, inputs[k].dtype) for k in outputs}
        out = jax.pure_callback(host, shapes, inputs, vmap_method="sequential")
    else:
        out = {k: jnp.asarray(v) for k, v in host(inputs).items()}
    if coupled:
        world.metabolites = out.pop("metabolites")
        world.concentration = out.pop("concentration")
    for name, value in out.items():
        setattr(pop, name, value)

