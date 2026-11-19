"""Measure the flow-head policy's flow-estimate error under live-fluid coupling.

For each condition and seed the recorded flow-head protocol is trained once (cached), then
the frozen genomes are evaluated on the held-out diagonals while recording, per step and
per agent, the angle between the policy's estimate ``unit(previous_velocity -
previous_thrust)`` and the live local flow it is scored against (see
``flow_estimate_errors``). A second phase evaluates genomes trained in one condition under
another condition's fluid dynamics to separate training-time from evaluation-time effects.

Usage:
    python scripts/measure_flow_estimate_error.py --out DIR --workers 8
    python scripts/measure_flow_estimate_error.py --out DIR --workers 8 --phase cross
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
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    embodied_law_config,
    evaluate_frozen_alignment,
    flow_estimate_errors,
    run_embodied_law_pilot,
)

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
DIAGONALS = ((0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071))
EVAL_STEPS = 12
PROTOCOL = dict(
    generations=8, population=16, steps_per_generation=8, size=14, proprioception=True,
    displacement_reward=5.0, mutate_survivors=False, linear_policy=False,
    unit_flow_features=True, fitness_mode="min_alignment", fitness_steps=6,
    rank_replacement=True, immigrant_fraction=0.25, flow_head=True, raw_flow_features=False,
)


def _live(drag: float, coupling: str) -> dict:
    return {
        "flow_solver": "lbm_live", "lbm_box": "periodic", "lbm_peak_speed": 0.25,
        "lbm_law_relax_steps": 20.0, "lbm_agent_drag": drag, "lbm_agent_coupling": coupling,
    }


CONDITIONS = {
    "A": None,  # legacy uniform law
    "B": _live(0.0, "monopole"),
    "C": _live(0.5, "monopole"),
    "D": _live(0.5, "dipole"),
    "E": _live(2.0, "monopole"),
    "F": _live(2.0, "dipole"),
}
SEEDS = tuple(range(15))


def law_config(seed: int, overrides: dict | None):
    kw = {k: v for k, v in PROTOCOL.items() if k not in ("fitness_mode", "fitness_steps")}
    config = embodied_law_config(seed, **kw)
    return replace(config, world=replace(config.world, **overrides)) if overrides else config


def _to_numpy(genomes) -> dict:
    return {k: np.asarray(v) for k, v in genomes.items()}


def evaluate(config, genomes) -> dict:
    """Per-diagonal alignment and estimate-error summaries for one genome set."""
    out = {}
    for direction in DIAGONALS:
        records: list = []
        result = evaluate_frozen_alignment(
            None, config, direction, steps=EVAL_STEPS, genomes=genomes, diagnostics=records
        )
        out[str(direction)] = {**result, **flow_estimate_errors(records)}
    return out


def summarise(evolved: dict, initial: dict) -> dict:
    keys = list(evolved)
    deltas = {k: evolved[k]["mean_alignment"] - initial[k]["mean_alignment"] for k in keys}
    err = [evolved[k]["angle_mean"] for k in keys]
    return {
        "min_delta": float(min(deltas.values())),
        "delta": deltas,
        "angle_mean": float(np.nanmean(err)),
        "angle_max_direction": float(np.nanmax(err)),
        "law_angle_mean": float(np.nanmean([evolved[k]["law_angle_mean"] for k in keys])),
        "magnitude_dev_mean": float(np.nanmean([evolved[k]["magnitude_dev_mean"] for k in keys])),
        "frac_over_45": float(np.nanmean([evolved[k]["frac_over_45"] for k in keys])),
        "initial_angle_mean": float(np.nanmean([initial[k]["angle_mean"] for k in keys])),
        "evolved": evolved,
        "initial": initial,
    }


def train_job(args) -> str:
    cond, seed, out = args
    cache = Path(out) / "cache" / f"{cond}_{seed}.pkl"
    if cache.exists():
        return f"cached {cond} {seed}"
    pilot = run_embodied_law_pilot(
        seed=seed, directions=CARDINALS, world_overrides=CONDITIONS[cond], **PROTOCOL
    )
    config = pilot["config"]
    genomes = _to_numpy(pilot["state"].embodied.genomes)
    initial = _to_numpy(pilot["initial_genomes"])
    row = summarise(evaluate(config, genomes), evaluate(config, initial))
    row.update(cond=cond, seed=seed)
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump({"genomes": genomes, "initial": initial, "row": row}, fh)
    return f"done {cond} {seed} min_delta={row['min_delta']:.3f} angle={row['angle_mean']:.1f}"


CROSS = (("B", "E"), ("E", "B"), ("B", "F"), ("F", "B"), ("E", "F"), ("F", "E"))


def cross_job(args) -> str:
    train, test, seed, out = args
    path = Path(out) / "cross" / f"train{train}_eval{test}_{seed}.json"
    if path.exists():
        return f"cached {train}->{test} {seed}"
    load = lambda c: pickle.load(open(Path(out) / "cache" / f"{c}_{seed}.pkl", "rb"))  # noqa: E731
    src, ref = load(train), load("B")
    config = law_config(seed, CONDITIONS[test])
    same_initial = all(np.array_equal(src["initial"][k], ref["initial"][k]) for k in src["initial"])
    row = summarise(evaluate(config, src["genomes"]), evaluate(config, src["initial"]))
    row.update(train=train, eval=test, seed=seed, initial_genomes_identical_to_B=bool(same_initial))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(row, indent=1, sort_keys=True))
    return f"done {train}->{test} {seed} min_delta={row['min_delta']:.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--phase", choices=("train", "cross"), default="train")
    parser.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    args = parser.parse_args()
    if args.phase == "train":
        jobs = [(c, s, str(args.out)) for s in args.seeds for c in args.conditions]
        fn = train_job
    else:
        jobs = [(t, e, s, str(args.out)) for s in args.seeds for t, e in CROSS]
        fn = cross_job
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        for line in pool.imap_unordered(fn, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
