#!/usr/bin/env python
"""Pilot: legacy vs physics-gene POET at equal budget, cross-evaluated on both held-out suites."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from hybrid_alife import poet

ARMS = ("legacy", "physics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poet_physics_pilot.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument("--out", default="docs/results/poet_physics_envs/pilot.json")
    parser.add_argument("--artifacts", default=None, help="keep full run artifacts here")
    args = parser.parse_args()

    base = poet.load_poet_config(args.config, track="poet")
    configs = {arm: replace(base, physics_genes=arm == "physics", condition=arm) for arm in ARMS}
    artifacts = Path(args.artifacts or tempfile.mkdtemp(prefix="poet-physics-"))
    cells = []
    for seed in args.seeds:
        # Both suites are fixed per seed and never used for training or selection.
        suites = {arm: poet.initialize_heldout_environments(configs[arm], seed) for arm in ARMS}
        for arm in ARMS:
            cfg = configs[arm]
            start = time.perf_counter()
            state = poet.run_poet(
                cfg, seed, generations=args.generations, out_dir=artifacts / arm / f"seed-{seed}"
            )
            seconds = time.perf_counter() - start
            initial = poet.initialize_poet_state(cfg, seed).policies
            heldout = {}
            for suite, environments in suites.items():
                scores = {}
                for label, policies in (("initial", initial), ("final", state.policies)):
                    matrix = poet.evaluate_transfer_matrix(
                        policies, environments, configs[suite], seed + 900_000
                    )
                    # Best policy per environment is an optimistic, selection-on-test score.
                    scores[f"{label}_mean"] = float(matrix.mean())
                    scores[f"{label}_best_per_env_mean"] = float(matrix.max(axis=0).mean())
                heldout[suite] = scores
            cell = {
                "arm": arm,
                "seed": seed,
                "heldout": heldout,
                "policy_updates": state.policy_updates,
                "environment_candidates": state.environment_candidates,
                "environment_admissions": state.environment_admissions,
                "candidate_score_mean": float(
                    np.mean([row["candidate_score"] for row in state.admission_diagnostics])
                ),
                "ecology": state.ecology_diagnostics,
                "wall_seconds": round(seconds, 1),
            }
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    result = {
        "status": "pilot",
        "source_commit": poet._source_commit(),
        "config": args.config,
        "seeds": args.seeds,
        "generations": args.generations,
        "cells": cells,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}; run artifacts in {artifacts}")


if __name__ == "__main__":
    main()
