"""Bounded policy-gradient study for the embodied latent law.

Trains one embodied genome with REINFORCE on the four cardinals, then evaluates
frozen on the held-out diagonals against the initial-policy and reversed-law
controls. The gate is the same as the evolution pilots.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_embodied_policy_gradient.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    embodied_law_config,
    per_direction_frozen_alignment,
)
from hybrid_alife.experiments.latent_law_embodied_pg import (  # noqa: E402
    single_agent_config,
    train_policy_gradient,
)

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
DIAGONALS = ((0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071))

GATE_MARGIN = 0.2
GATE_REVERSED_MIN = 0.2


def study_seed(
    seed: int,
    *,
    updates: int,
    episodes_per_update: int,
    steps: int,
    size: int,
    learning_rate: float,
    sigma: float,
    eval_steps: int,
    random_directions: bool,
) -> dict:
    base = embodied_law_config(
        seed,
        generations=1,
        population=2,
        steps_per_generation=steps,
        size=size,
        proprioception=True,
    )
    trained, initial = train_policy_gradient(
        base,
        directions=CARDINALS,
        updates=updates,
        episodes_per_update=episodes_per_update,
        steps=steps,
        learning_rate=learning_rate,
        sigma=sigma,
        random_directions=random_directions,
    )
    eval_config = single_agent_config(base)
    evolved = per_direction_frozen_alignment(
        None, eval_config, DIAGONALS, steps=eval_steps, genomes=trained
    )
    initial_eval = per_direction_frozen_alignment(
        None, eval_config, DIAGONALS, steps=eval_steps, genomes=initial
    )
    reversed_law = per_direction_frozen_alignment(
        None, eval_config, DIAGONALS, steps=eval_steps, genomes=trained, reverse_law=True
    )
    deltas = {
        key: evolved[key]["mean_alignment"] - initial_eval[key]["mean_alignment"]
        for key in evolved
    }
    passed = all(
        deltas[key] >= GATE_MARGIN
        and reversed_law[key]["mean_alignment"] >= GATE_REVERSED_MIN
        for key in evolved
    )
    return {
        "seed": seed,
        "passed": bool(passed),
        "trained": {k: v["mean_alignment"] for k, v in evolved.items()},
        "initial": {k: v["mean_alignment"] for k, v in initial_eval.items()},
        "reversed_law": {k: v["mean_alignment"] for k, v in reversed_law.items()},
        "delta": deltas,
        "min_delta": float(min(deltas.values())),
        "min_reversed": float(min(v["mean_alignment"] for v in reversed_law.values())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--updates", type=int, default=60)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--sigma", type=float, default=0.4)
    parser.add_argument("--random-directions", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    results = []
    for seed in args.seeds:
        row = study_seed(
            seed,
            updates=args.updates,
            episodes_per_update=args.episodes_per_update,
            steps=args.steps,
            size=args.size,
            learning_rate=args.learning_rate,
            sigma=args.sigma,
            eval_steps=args.eval_steps,
            random_directions=args.random_directions,
        )
        results.append(row)
        print(
            f"seed {seed}: passed={row['passed']} "
            f"min_delta={row['min_delta']:.3f} min_reversed={row['min_reversed']:.3f}",
            flush=True,
        )

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "status": "pilot_only",
        "seeds": args.seeds,
        "updates": args.updates,
        "episodes_per_update": args.episodes_per_update,
        "steps": args.steps,
        "size": args.size,
        "eval_steps": args.eval_steps,
        "learning_rate": args.learning_rate,
        "sigma": args.sigma,
        "random_directions": args.random_directions,
        "gate": {"margin": GATE_MARGIN, "reversed_min": GATE_REVERSED_MIN},
        "seeds_passed": passed,
        "seeds_total": len(results),
        "results": results,
    }
    print(
        f"\n{passed}/{len(results)} seeds pass "
        f"(margin >= {GATE_MARGIN}, reversed >= {GATE_REVERSED_MIN})"
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
