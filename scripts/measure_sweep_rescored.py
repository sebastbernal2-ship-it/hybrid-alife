"""Rescore the monopole drag sweep against targets the agents cannot influence.

For each drag D, genomes trained at D (retrained deterministically, or copied from the
mechanism study's caches for drag 0 and 2.0) and the shared initial genomes are rolled out on
the four held-out diagonals in the fluid at D with one shadow fluid per agent, then rescored
against the official local flow, the imposed law, the flow without the agent's own wake, and
law plus own wake only (see ``measure_monopole_mechanism``). The D-trained genomes are also
evaluated in the uncoupled fluid, which removes every evaluation-time effect.

Usage:
    python scripts/measure_sweep_rescored.py train --out DIR --workers 12 [--seed-caches DIR]
    python scripts/measure_sweep_rescored.py eval  --out DIR --workers 12
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
import shutil  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from measure_flow_estimate_error import (  # noqa: E402
    CARDINALS,
    DIAGONALS,
    PROTOCOL,
    _live,
    _to_numpy,
    law_config,
)
from measure_monopole_mechanism import perturbation_stats, rescore, rollout  # noqa: E402

from hybrid_alife.experiments.latent_law_embodied import run_embodied_law_pilot  # noqa: E402

SEEDS = tuple(range(15))
DRAGS = (0.0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)  # monopole drag; 0 is the uncoupled live fluid


def overrides(drag: float) -> dict:
    return _live(drag, "monopole")


def cache_path(out, drag: float, seed: int) -> Path:
    return Path(out) / "cache" / f"M{drag}_{seed}.pkl"


def train_job(args) -> str:
    drag, seed, out = args
    path = cache_path(out, drag, seed)
    if path.exists():
        return f"cached drag {drag} seed {seed}"
    pilot = run_embodied_law_pilot(
        seed=seed, directions=CARDINALS, world_overrides=overrides(drag), **PROTOCOL
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "genomes": _to_numpy(pilot["state"].embodied.genomes),
                "initial": _to_numpy(pilot["initial_genomes"]),
            },
            fh,
        )
    return f"trained drag {drag} seed {seed}"


def _load(out, drag, seed):
    with open(cache_path(out, drag, seed), "rb") as fh:
        return pickle.load(fh)


def eval_job(args) -> str:
    seed, out = args
    path = Path(out) / "eval" / f"seed{seed}.json"
    if path.exists():
        return f"cached eval {seed}"
    base = _load(out, 0.0, seed)
    initial = base["initial"]
    config_b = law_config(seed, overrides(0.0))
    rows: list[dict] = []

    def add(kind, drag, label, genomes, config, coupled):
        for direction in DIAGONALS:
            recs = rollout(config, genomes, direction, shadows=coupled)
            rows.append(
                {
                    "kind": kind,
                    "drag": drag,
                    "genomes": label,
                    "direction": str(direction),
                    **rescore(recs),
                    **perturbation_stats(recs),
                }
            )

    # Uncoupled reference: drag-0 genomes and the initial genomes in the uncoupled fluid.
    add("own", 0.0, "trained", base["genomes"], config_b, False)
    add("own", 0.0, "init", initial, config_b, False)
    for drag in DRAGS[1:]:
        cached = _load(out, drag, seed)
        same = all(np.array_equal(initial[k], cached["initial"][k]) for k in initial)
        assert same, "initial genomes must be identical across conditions for a seed"
        config_d = law_config(seed, overrides(drag))
        add("own", drag, "trained", cached["genomes"], config_d, True)
        add("own", drag, "init", initial, config_d, True)
        add("transfer", drag, "trained", cached["genomes"], config_b, False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"seed": seed, "rows": rows}, indent=1, sort_keys=True))
    return f"eval {seed}"


def seed_caches(out: Path, source: Path) -> None:
    """Copy the mechanism study's B (drag 0) and E (drag 2.0) caches, which are identical runs."""
    for seed in SEEDS:
        for name, drag in (("B", 0.0), ("E", 2.0)):
            src = source / f"{name}_{seed}.pkl"
            dst = cache_path(out, drag, seed)
            if src.exists() and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train", "eval"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--seed-caches", type=Path, default=None)
    args = parser.parse_args()
    out = str(args.out)
    if args.seed_caches is not None:
        seed_caches(args.out, args.seed_caches)
    if args.phase == "train":
        jobs = [(d, s, out) for s in args.seeds for d in DRAGS]
        fn = train_job
    else:
        jobs, fn = [(s, out) for s in args.seeds], eval_job
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for line in pool.imap_unordered(fn, jobs):
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
