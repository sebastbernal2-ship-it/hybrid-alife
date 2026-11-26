"""Two follow-ups to the live-fluid law study.

``speed``: retrain and re-evaluate the drag-2.0 arms (E monopole, F dipole) exactly as the
study did, recording the largest lattice speed after *every* LBM sub-step. ``LiveLBM.advance``
only checks the 0.3 guard after a whole world step, so the sub-steps are measured by also
running 1..N-1 sub-steps from the same input (the trajectory itself is unchanged). The
retrained ``min_delta`` must equal the committed one.

``masked``: test the own-wake-removed (``noself``) target, which was estimated by shadow-fluid
superposition. For each agent i the fluid is rerun with agent i's force masked out:
``open`` replays every other agent's recorded force along the unchanged trajectories (the exact
counterpart of the estimate), ``closed`` reruns the whole simulation with agent i not forcing.

Usage:
    python scripts/measure_lbm_live_followups.py speed --out DIR --workers 2
    python scripts/measure_lbm_live_followups.py masked --out DIR --cache CACHE --seeds 0 1 2
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
import sys  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import measure_monopole_mechanism as mech  # noqa: E402
import run_embodied_law_study as study  # noqa: E402
from measure_flow_estimate_error import CONDITIONS, DIAGONALS, PROTOCOL, law_config  # noqa: E402

from hybrid_alife.experiments import runner  # noqa: E402
from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    _impose_flow,
    signed_alignment,
)
from hybrid_alife.physics.lbm import MAX_LATTICE_SPEED, LiveLBM  # noqa: E402

LIVE = ROOT / "docs" / "results" / "embodied_law_lbm_live"
COMMITTED = {"E": "seeds0-14_E_mono_2.0.json", "F": "seeds0-14_F_dipole_2.0.json"}


@contextmanager
def _speed_probe(peaks: dict):
    """Record the max lattice speed over every sub-step into ``peaks[peaks["phase"]]``."""
    original = LiveLBM.advance
    original_eval = study.per_direction_frozen_alignment

    def advance(self, steps, force=None):
        out = self
        for k in range(1, steps + 1):
            out = original(self, k, force)  # raises if the guard is crossed at sub-step k
            speed = float(np.hypot(out.u, out.v).max())
            peaks[peaks["phase"]] = max(peaks.get(peaks["phase"], 0.0), speed)
            if k == steps:
                peaks["end_of_step"] = max(peaks.get("end_of_step", 0.0), speed)
        return out if steps else original(self, steps, force)

    def frozen(*args, **kwargs):
        peaks["phase"] = "evaluation"
        return original_eval(*args, **kwargs)

    LiveLBM.advance, study.per_direction_frozen_alignment = advance, frozen
    try:
        yield
    finally:
        LiveLBM.advance, study.per_direction_frozen_alignment = original, original_eval


def speed_job(args) -> str:
    cond, seed, out = args
    path = Path(out) / f"{cond}_{seed}.json"
    if path.exists():
        return f"cached {cond} {seed}"
    peaks: dict = {"phase": "training"}
    with _speed_probe(peaks):
        row = study.study_seed(
            seed, **PROTOCOL, eval_steps=12, min_evolved=None, world_overrides=CONDITIONS[cond]
        )
    committed = {r["seed"]: r for r in json.loads((LIVE / COMMITTED[cond]).read_text())}
    result = {
        "cond": cond, "seed": seed, "training": peaks["training"],
        "evaluation": peaks["evaluation"], "end_of_step": peaks["end_of_step"],
        "min_delta": row["min_delta"], "committed_min_delta": committed[seed]["min_delta"],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=1, sort_keys=True))
    return (f"{cond} {seed} train={peaks['training']:.4f} eval={peaks['evaluation']:.4f} "
            f"guard={MAX_LATTICE_SPEED} diff={row['min_delta'] - committed[seed]['min_delta']:.2e}")


def _cell_flow(field: np.ndarray, positions) -> np.ndarray:
    h, w = field.shape[:2]
    cells = np.clip(np.floor(np.asarray(positions) * np.array([h, w])).astype(int),
                    0, [h - 1, w - 1])
    return field[cells[:, 0], cells[:, 1]]


def masked_rollout(config, genomes, direction, *, mask_agent: int | None):
    """Frozen coupled rollout. With ``mask_agent`` set, that agent never forces the fluid
    (closed loop). Otherwise one extra fluid per agent i is forced by everyone but i along the
    unchanged trajectories (open loop). Returns per-step records."""
    eval_config = replace(config, embodied=replace(config.embodied, initial_alive_fraction=1.0))
    target = runner.initialize_sim(eval_config)
    target.embodied.genomes = {k: jnp.asarray(v) for k, v in genomes.items()}
    target.avida = None
    target = _impose_flow(target, tuple(float(v) for v in direction), mech.MAGNITUDE)
    n = target.embodied.alive.shape[0]
    h, w = target.world.flow.shape[:2]
    open_loop = mask_agent is None
    ctx: dict = {"fluids": [target.lbm] * n, "forces": [np.zeros((h, w, 2))] * n}
    orig_world, orig_couple = runner.step_world, runner._couple_agents_to_fluid

    def step_world(state, cfg):
        state = orig_world(state, cfg)
        if open_loop:
            ctx["fluids"] = [
                fluid.advance(cfg.lbm_substeps, force=force)
                for fluid, force in zip(ctx["fluids"], ctx["forces"], strict=True)
            ]
        return state

    def couple(state, cfg, old_positions):
        alive = np.asarray(state.embodied.alive)
        if open_loop:
            orig_couple(state, cfg, old_positions)
            own = [
                runner.agent_force_field(
                    state, cfg, old_positions, alive=alive & (np.arange(n) == i))
                for i in range(n)
            ]
            ctx["forces"] = [state.lbm.pending_force - f for f in own]
        else:
            state.lbm = state.lbm.with_pending_force(runner.agent_force_field(
                state, cfg, old_positions, alive=alive & (np.arange(n) != mask_agent)))

    records, lineage = [], config.embodied.population_size
    runner.step_world, runner._couple_agents_to_fluid = step_world, couple
    try:
        for _ in range(mech.EVAL_STEPS):
            target, lineage = runner.step_sim(
                target, eval_config, lineage, allow_reproduction=False)
            pop = target.embodied
            rec = {
                "thrust": np.asarray(pop.last_action[:, :2], np.float64),
                "alive": np.asarray(pop.alive, bool),
                "flow": _cell_flow(np.asarray(target.world.flow, np.float64), pop.positions),
            }
            if open_loop:
                rec["masked"] = np.stack([
                    _cell_flow(ctx["fluids"][i].flow().astype(np.float64), pop.positions)[i]
                    for i in range(n)
                ])
            records.append(rec)
    finally:
        runner.step_world, runner._couple_agents_to_fluid = orig_world, orig_couple
    return records


def masked_direction(config, genomes, direction) -> dict:
    estimate = mech.rollout(config, genomes, direction, shadows=True)
    exact = masked_rollout(config, genomes, direction, mask_agent=None)
    n = exact[0]["alive"].shape[0]
    errors, wakes, super_scores, open_scores = [], [], [], []
    for est, rec in zip(estimate, exact, strict=True):
        np.testing.assert_allclose(est.flow, rec["flow"], atol=1e-7)  # same trajectory
        superposed = est.flow - est.self_pert
        ok = rec["alive"]
        errors.append(np.linalg.norm(superposed - rec["masked"], axis=-1)[ok])
        wakes.append(np.linalg.norm(rec["flow"] - rec["masked"], axis=-1)[ok])
        super_scores.append(signed_alignment(rec["thrust"], superposed, ok))
        open_scores.append(signed_alignment(rec["thrust"], rec["masked"], ok))
    # Closed loop: agent i's alignment with its own local flow when it never forces.
    closed = [[] for _ in range(mech.EVAL_STEPS)]
    for i in range(n):
        for t, rec in enumerate(masked_rollout(config, genomes, direction, mask_agent=i)):
            one = rec["alive"] & (np.arange(n) == i)
            if one.any():
                closed[t].append(signed_alignment(rec["thrust"], rec["flow"], one))
    errors, wakes = np.concatenate(errors), np.concatenate(wakes)
    return {
        "official": mech.rescore(estimate)["official"],
        "noself_superposition": float(np.mean(super_scores)),
        "noself_open": float(np.mean(open_scores)),
        "noself_closed": float(np.mean([np.mean(step) if step else 0.0 for step in closed])),
        "target_error_mean": float(errors.mean()), "target_error_max": float(errors.max()),
        "own_wake_mean": float(wakes.mean()),
    }


def masked_job(args) -> str:
    seed, out, cache = args
    path = Path(out) / f"seed{seed}.json"
    if path.exists():
        return f"cached masked {seed}"
    sets = mech._genome_sets(cache, seed)
    rows = []
    for fluid in ("B", "E"):
        config = law_config(seed, CONDITIONS[fluid])
        for name, genomes in sets.items():
            for direction in DIAGONALS:
                if fluid == "B":  # uncoupled reference: every target is the law
                    value = mech.rescore(
                        mech.rollout(config, genomes, direction, shadows=False))["official"]
                    row = {"official": value}
                else:
                    row = masked_direction(config, genomes, direction)
                rows.append({"fluid": fluid, "genomes": name, "direction": str(direction), **row})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "rows": rows}, indent=1, sort_keys=True))
    return f"masked {seed}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("speed", "masked"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="directory holding cache/{B,E}_{seed}.pkl")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(15)))
    args = parser.parse_args()
    if args.phase == "speed":
        jobs, fn = [(c, s, str(args.out)) for s in args.seeds for c in ("E", "F")], speed_job
    else:
        if args.cache is None:
            parser.error("masked needs --cache (see measure_monopole_mechanism.py train)")
        jobs = [(s, str(args.out), str(args.cache)) for s in args.seeds]
        fn = masked_job
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(fn, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
