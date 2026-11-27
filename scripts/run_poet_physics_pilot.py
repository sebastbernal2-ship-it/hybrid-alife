#!/usr/bin/env python
"""Pilot: legacy vs physics-gene POET at equal budget, cross-evaluated on both held-out suites."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np

from hybrid_alife import poet

ARMS = ("legacy", "physics")
CALIBRATION_QUANTILE = 0.25  # share of candidate worlds the criterion should reject


def policy_behaviour(policies, environments, cfg, seed: int) -> dict[str, list[float]]:
    """Per policy: greedy modal-action share and mean action entropy (nats), pooled over worlds."""
    modal, entropy = [], []
    for policy in policies:
        batches = [
            poet.collect_rollout(
                policy, env, cfg, jax.random.PRNGKey(seed + index),
                stochastic=False, horizon=cfg.transfer_steps,
            )
            for index, env in enumerate(environments)
        ]
        actions = np.concatenate([np.asarray(b.actions) for b in batches])
        observations = np.concatenate([np.asarray(b.observations) for b in batches])
        log_p = np.asarray(jax.nn.log_softmax(poet.policy_forward(policy, observations)[0]))
        modal.append(float(np.bincount(actions).max() / actions.size))
        entropy.append(float(-(np.exp(log_p) * log_p).sum(axis=1).mean()))
    return {"greedy_modal_action_share": modal, "entropy_nats": entropy}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/poet_physics_pilot.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--generations", type=int, default=12)
    parser.add_argument(
        "--calibrate", action="store_true",
        help="print a minimal_criterion from these seeds' candidate scores, admit-all; "
             "use seeds disjoint from the pilot's",
    )
    parser.add_argument("--out", default="docs/results/poet_physics_envs/pilot.json")
    parser.add_argument("--artifacts", default=None, help="keep full run artifacts here")
    args = parser.parse_args()

    base = poet.load_poet_config(args.config, track="poet")
    if args.calibrate:
        base = replace(base, minimal_criterion=-10.0)
    scores_seen: list[float] = []
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
            candidates = [row["candidate_score"] for row in state.admission_diagnostics]
            scores_seen += candidates
            if args.calibrate:
                continue
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
                    scores[f"{label}_behaviour"] = policy_behaviour(
                        policies, environments, configs[suite], seed + 900_000
                    )
                    scores[f"{label}_best_per_env_mean"] = float(matrix.max(axis=0).mean())
                heldout[suite] = scores
            cell = {
                "arm": arm,
                "seed": seed,
                "heldout": heldout,
                "policy_updates": state.policy_updates,
                "environment_candidates": state.environment_candidates,
                "environment_admissions": state.environment_admissions,
                "candidate_score_mean": float(np.mean(candidates)),
                "ecology": state.ecology_diagnostics,
                "wall_seconds": round(seconds, 1),
            }
            cells.append(cell)
            print(json.dumps(cell), flush=True)
    if args.calibrate:
        quantiles = np.quantile(scores_seen, [0, 0.25, 0.5, 0.75, 1])
        print(json.dumps({
            "calibration_seeds": args.seeds, "candidates": len(scores_seen),
            "min_q25_median_q75_max": quantiles.round(4).tolist(),
            "minimal_criterion": round(float(np.quantile(scores_seen, CALIBRATION_QUANTILE)), 2),
        }))
        return
    result = {
        "status": "pilot",
        "source_commit": poet._source_commit(),
        "config": args.config,
        "minimal_criterion": base.minimal_criterion,
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
