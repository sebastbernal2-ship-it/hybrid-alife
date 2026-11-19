"""Isolate which feature of the monopole-2.0 fluid lowers the measured alignment.

Frozen genomes are rolled out on the four held-out diagonals in the B (no coupling),
E (monopole 2.0) and F (dipole 2.0) live fluids. During each coupled rollout, one *shadow*
fluid per agent is advanced in lockstep, forced only by that agent. The fluid is near-linear
at these Mach numbers, so an agent's own wake at its own cell is ``shadow_i - law`` and the
flow without self-contamination is ``flow - (shadow_i - law)``. The same recorded rollouts
are then rescored against different targets (official local flow, imposed law, flow without
own wake, law plus own wake only), which separates target effects from dynamics effects.
A second control replays a recorded flow time series with the agents not forcing.

Usage:
    python scripts/measure_monopole_mechanism.py train --out DIR --workers 15
    python scripts/measure_monopole_mechanism.py eval  --out DIR --workers 15
    python scripts/measure_monopole_mechanism.py replay --out DIR --workers 15
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
)

import argparse  # noqa: E402
import json  # noqa: E402
import multiprocessing as mp  # noqa: E402
import pickle  # noqa: E402
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import dataclass, replace  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_flow_estimate_error import (  # noqa: E402
    CARDINALS,
    CONDITIONS,
    DIAGONALS,
    EVAL_STEPS,
    PROTOCOL,
    _to_numpy,
    law_config,
)

import hybrid_alife.experiments.runner as runner  # noqa: E402
from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    _impose_flow,
    run_embodied_law_pilot,
    signed_alignment,
)
from hybrid_alife.experiments.runner import (  # noqa: E402
    agent_force_field,
    initialize_sim,
    step_sim,
)

MAGNITUDE = 0.2
SEEDS = tuple(range(15))
FLUIDS = ("B", "E", "F")  # no coupling, monopole 2.0, dipole 2.0
GENOMES = ("B", "E", "init")  # B-trained, E-trained, shared initial genomes
VARIANTS = ("official", "law", "noself", "selfonly", "noself_sum")


# ---------------------------------------------------------------------------------------
# Rollouts with shadow fluids
# ---------------------------------------------------------------------------------------


@dataclass
class Record:
    """One step: thrust, aliveness, flow at each agent's cell, and the decomposition."""

    thrust: np.ndarray  # [N, 2]
    alive: np.ndarray  # [N]
    flow: np.ndarray  # [N, 2] official local flow
    self_pert: np.ndarray  # [N, 2] agent i's own wake at its own cell
    others_sum: np.ndarray  # [N, 2] sum of the other agents' shadow wakes at agent i's cell
    law: np.ndarray  # [2]
    superposition_residual: float  # ||sum_i pert_i - pert_full|| / ||pert_full||
    full_flow: np.ndarray | None = None  # [H, W, 2] when capture_fields


@contextmanager
def _shadow_fluids(ctx: dict):
    """Advance one shadow fluid per agent in lockstep with the main fluid."""
    orig_world, orig_couple = runner.step_world, runner._couple_agents_to_fluid

    def step_world(state, cfg):
        state = orig_world(state, cfg)
        if ctx.get("shadows") is not None:
            for i, shadow in enumerate(ctx["shadows"]):
                ctx["shadows"][i] = shadow.advance(cfg.lbm_substeps, force=ctx["forces"][i])
            ctx["snap"] = np.stack([s.flow() for s in ctx["shadows"]]).astype(np.float64)
        if ctx.get("capture") is not None:
            ctx["capture"].append(np.asarray(state.world.flow))
        return state

    def couple(state, cfg, old_positions):
        orig_couple(state, cfg, old_positions)
        n = state.embodied.alive.shape[0]
        alive = np.asarray(state.embodied.alive)
        ctx["forces"] = [
            agent_force_field(state, cfg, old_positions, alive=alive & (np.arange(n) == i))
            for i in range(n)
        ]
        total = sum(ctx["forces"])
        np.testing.assert_allclose(total, state.lbm.pending_force, atol=1e-12)

    runner.step_world, runner._couple_agents_to_fluid = step_world, couple
    try:
        yield
    finally:
        runner.step_world, runner._couple_agents_to_fluid = orig_world, orig_couple


def rollout(config, genomes: dict, direction, *, shadows: bool, capture_fields: bool = False,
            replay_series: list | None = None) -> list[Record]:
    """Frozen rollout recording per-step data. ``replay_series`` swaps in a recorded fluid."""
    eval_config = replace(
        config, embodied=replace(config.embodied, initial_alive_fraction=1.0)
    )
    target = initialize_sim(eval_config)
    target.embodied.genomes = {k: jnp.asarray(v) for k, v in genomes.items()}
    target.avida = None
    applied = np.asarray(direction, dtype=np.float64)
    target = _impose_flow(target, tuple(float(v) for v in applied), MAGNITUDE)
    law = applied / np.linalg.norm(applied) * MAGNITUDE
    n = target.embodied.alive.shape[0]
    h, w = target.world.flow.shape[:2]
    ctx: dict = {"capture": [] if capture_fields else None}
    if shadows:
        ctx["shadows"] = [target.lbm] * n
        ctx["forces"] = [np.zeros((h, w, 2))] * n
        ctx["snap"] = None
    if replay_series is not None:
        target.lbm = _Replay(replay_series)
    records: list[Record] = []
    lineage = config.embodied.population_size
    with _shadow_fluids(ctx):
        for _ in range(EVAL_STEPS):
            target, lineage = step_sim(target, eval_config, lineage, allow_reproduction=False)
            pop = target.embodied
            flow_field = np.asarray(target.world.flow, dtype=np.float64)
            cells = np.clip(
                np.floor(np.asarray(pop.positions) * np.array([h, w])).astype(int),
                0, [h - 1, w - 1],
            )
            flow = flow_field[cells[:, 0], cells[:, 1]]
            if ctx.get("snap") is not None:
                snap = ctx["snap"]  # [N, H, W, 2] shadow flows after this step's advance
                pert_all = snap - law
                self_pert = pert_all[np.arange(n), cells[:, 0], cells[:, 1]]
                total = pert_all.sum(axis=0)
                others_sum = total[cells[:, 0], cells[:, 1]] - self_pert
                full = flow_field - law
                denom = np.linalg.norm(full)
                # Steps with (almost) no wake are float noise, not a superposition test.
                residual = float(np.linalg.norm(total - full) / denom) if denom > 1e-2 else 0.0
            else:
                self_pert, others_sum, residual = (np.zeros_like(flow),) * 2 + (0.0,)
            records.append(Record(
                thrust=np.asarray(pop.last_action[:, :2], dtype=np.float64),
                alive=np.asarray(pop.alive, dtype=bool), flow=flow, self_pert=self_pert,
                others_sum=others_sum,
                law=law, superposition_residual=residual,
                full_flow=flow_field if capture_fields else None,
            ))
    return records


class _Replay:
    """Minimal stand-in for LiveLBM that replays a recorded per-step flow series."""

    def __init__(self, series: list, index: int = 0):
        self.series, self.index = series, index
        self.mask = np.zeros((1, 1), np.uint8)
        self.target = None
        self.peak_speed = 1.0
        self.pending_force = None

    def advance(self, steps: int, force=None):
        return _Replay(self.series, self.index + 1)

    def flow(self):
        return np.asarray(self.series[min(self.index - 1, len(self.series) - 1)], np.float32)

    def with_pending_force(self, force):
        return self


def rescore(records: list[Record]) -> dict[str, float]:
    """Mean per-step signed alignment of the thrust against each target variant."""
    out = {}
    for variant in VARIANTS:
        values = []
        for rec in records:
            if variant == "official":
                target = rec.flow
            elif variant == "law":
                target = np.broadcast_to(rec.law, rec.flow.shape)
            elif variant == "noself":
                target = rec.flow - rec.self_pert
            elif variant == "noself_sum":
                target = rec.law + rec.others_sum
            else:
                target = rec.law + rec.self_pert
            values.append(signed_alignment(rec.thrust, target, rec.alive))
        out[variant] = float(np.mean(values))
    return out


def perturbation_stats(records: list[Record]) -> dict[str, float]:
    """Size of the wake the agents see, and how much of it is their own."""
    own, full, cos_own, others = [], [], [], []
    for rec in records:
        ok = rec.alive
        own.append(np.linalg.norm(rec.self_pert[ok], axis=-1))
        others.append(np.linalg.norm(rec.others_sum[ok], axis=-1))
        full.append(np.linalg.norm(rec.flow[ok] - rec.law, axis=-1))
        t = np.linalg.norm(rec.thrust, axis=-1)
        s = np.linalg.norm(rec.self_pert, axis=-1)
        v = ok & (t > 1e-8) & (s > 1e-12)
        if v.any():
            cos_own.append(np.sum(rec.thrust[v] * rec.self_pert[v], axis=-1) / (t[v] * s[v]))
    cat = lambda xs: np.concatenate(xs) if xs else np.array([])  # noqa: E731
    return {
        "own_wake_mean": float(cat(own).mean()),
        "others_wake_mean": float(cat(others).mean()),
        "total_perturbation_mean": float(cat(full).mean()),
        "cos_thrust_own_wake": float(cat(cos_own).mean()) if cos_own else float("nan"),
        "superposition_residual_mean": float(np.mean([r.superposition_residual for r in records])),
        "superposition_residual_max": float(np.max([r.superposition_residual for r in records])),
    }


# ---------------------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------------------


def train_job(args) -> str:
    cond, seed, out = args
    path = Path(out) / "cache" / f"{cond}_{seed}.pkl"
    if path.exists():
        return f"cached {cond} {seed}"
    pilot = run_embodied_law_pilot(
        seed=seed, directions=CARDINALS, world_overrides=CONDITIONS[cond], **PROTOCOL
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump({"genomes": _to_numpy(pilot["state"].embodied.genomes),
                     "initial": _to_numpy(pilot["initial_genomes"])}, fh)
    return f"trained {cond} {seed}"


def _load(out, cond, seed):
    with open(Path(out) / "cache" / f"{cond}_{seed}.pkl", "rb") as fh:
        return pickle.load(fh)


def _genome_sets(out, seed) -> dict[str, dict]:
    b, e = _load(out, "B", seed), _load(out, "E", seed)
    same = all(np.array_equal(b["initial"][k], e["initial"][k]) for k in b["initial"])
    assert same, "initial genomes must be identical across conditions for a seed"
    return {"B": b["genomes"], "E": e["genomes"], "init": b["initial"]}


def eval_job(args) -> str:
    seed, out = args
    path = Path(out) / "eval" / f"seed{seed}.json"
    if path.exists():
        return f"cached eval {seed}"
    sets = _genome_sets(out, seed)
    result: dict = {"seed": seed, "rows": []}
    for fluid in FLUIDS:
        config = law_config(seed, CONDITIONS[fluid])
        coupled = CONDITIONS[fluid]["lbm_agent_drag"] > 0
        for gname, genomes in sets.items():
            for direction in DIAGONALS:
                recs = rollout(config, genomes, direction, shadows=coupled)
                result["rows"].append({
                    "fluid": fluid, "genomes": gname, "direction": str(direction),
                    **rescore(recs), **perturbation_stats(recs),
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    return f"eval {seed}"


def replay_job(args) -> str:
    """Cross-replay: each genome set is scored in the E-fluid series produced by each set.

    Replaying a population's own series is a tautology (its behaviour depends only on the flow
    it sees, so the trajectory repeats exactly), so the informative cells are the foreign ones:
    the series recorded from population S is replayed with the agents not forcing, and genome
    set G is scored in it. G == S carries G's own wake; G != S carries a foreign wake that G did
    not create and cannot influence.
    """
    seed, out = args
    path = Path(out) / "replay" / f"seed{seed}.json"
    if path.exists():
        return f"cached replay {seed}"
    sets = _genome_sets(out, seed)
    config_e = law_config(seed, CONDITIONS["E"])
    config_replay = law_config(seed, {**CONDITIONS["E"], "lbm_agent_drag": 0.0})
    result: dict = {"seed": seed, "rows": []}
    for direction in DIAGONALS:
        series = {}
        for source in ("B", "init"):
            full = rollout(config_e, sets[source], direction, shadows=False, capture_fields=True)
            series[source] = [r.full_flow for r in full]
        for source in ("B", "init"):
            for gname in ("B", "init"):
                rep = rollout(config_replay, sets[gname], direction, shadows=False,
                              replay_series=series[source])
                result["rows"].append({
                    "genomes": gname, "series_from": source, "direction": str(direction),
                    **rescore(rep),
                })
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    return f"replay {seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval", "replay"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    out = str(args.out)
    if args.phase == "train":
        jobs, fn = [(c, s, out) for s in args.seeds for c in ("B", "E")], train_job
    elif args.phase == "eval":
        jobs, fn = [(s, out) for s in args.seeds], eval_job
    else:
        jobs, fn = [(s, out) for s in args.seeds], replay_job
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(fn, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
