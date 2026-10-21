"""Bounded evolved-embodied law study with proprioception.

Runs the pilot with previous-velocity and previous-thrust features plus a
displacement reward, then evaluates frozen genomes on held-out diagonals against
two controls: the initial population and a reversed law.

Gate, per held-out direction: the evolved alignment must beat the initial
population by a margin, and the reversed-law alignment must be negative. The
minimum over directions decides, never an average across opposite directions.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hybrid_alife.experiments.latent_law_embodied import (  # noqa: E402
    per_direction_frozen_alignment,
    run_embodied_law_pilot,
)

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
DIAGONALS = ((0.7071, -0.7071), (0.7071, 0.7071), (-0.7071, 0.7071), (-0.7071, -0.7071))

GATE_MARGIN = 0.2
GATE_REVERSED_MIN = 0.2


def study_seed(
    seed: int,
    *,
    generations: int,
    population: int,
    steps_per_generation: int,
    size: int,
    proprioception: bool,
    displacement_reward: float,
    mutate_survivors: bool,
    linear_policy: bool,
    unit_flow_features: bool,
    fitness_mode: str,
    fitness_steps: int,
    rank_replacement: bool,
    immigrant_fraction: float,
    flow_head: bool,
    eval_steps: int,
) -> dict:
    pilot = run_embodied_law_pilot(
        seed=seed,
        directions=CARDINALS,
        generations=generations,
        population=population,
        steps_per_generation=steps_per_generation,
        size=size,
        proprioception=proprioception,
        displacement_reward=displacement_reward,
        mutate_survivors=mutate_survivors,
        linear_policy=linear_policy,
        unit_flow_features=unit_flow_features,
        fitness_mode=fitness_mode,
        fitness_steps=fitness_steps,
        rank_replacement=rank_replacement,
        immigrant_fraction=immigrant_fraction,
        flow_head=flow_head,
    )
    state, config = pilot["state"], pilot["config"]
    evolved = per_direction_frozen_alignment(state, config, DIAGONALS, steps=eval_steps)
    initial = per_direction_frozen_alignment(
        state, config, DIAGONALS, steps=eval_steps, genomes=pilot["initial_genomes"]
    )
    reversed_law = per_direction_frozen_alignment(
        state, config, DIAGONALS, steps=eval_steps, reverse_law=True
    )

    deltas = {
        key: evolved[key]["mean_alignment"] - initial[key]["mean_alignment"]
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
        "evolved": {k: v["mean_alignment"] for k, v in evolved.items()},
        "initial": {k: v["mean_alignment"] for k, v in initial.items()},
        "reversed_law": {k: v["mean_alignment"] for k, v in reversed_law.items()},
        "delta": deltas,
        "min_delta": float(min(deltas.values())),
        "min_reversed": float(min(v["mean_alignment"] for v in reversed_law.values())),
        "training_alignment": pilot["training_alignment"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--steps-per-generation", type=int, default=8)
    parser.add_argument("--size", type=int, default=14)
    parser.add_argument("--eval-steps", type=int, default=12)
    parser.add_argument("--displacement-reward", type=float, default=5.0)
    parser.add_argument("--no-proprioception", action="store_true")
    parser.add_argument("--mutate-survivors", action="store_true")
    parser.add_argument("--linear-policy", action="store_true")
    parser.add_argument("--unit-flow-features", action="store_true")
    parser.add_argument(
        "--fitness-mode", choices=("energy", "min_alignment"), default="energy"
    )
    parser.add_argument("--fitness-steps", type=int, default=6)
    parser.add_argument("--rank-replacement", action="store_true")
    parser.add_argument("--immigrant-fraction", type=float, default=0.0)
    parser.add_argument("--flow-head", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    conditions = [not args.no_proprioception]
    results = []
    for proprioception in conditions:
        for seed in args.seeds:
            row = study_seed(
                seed,
                generations=args.generations,
                population=args.population,
                steps_per_generation=args.steps_per_generation,
                size=args.size,
                proprioception=proprioception,
                displacement_reward=args.displacement_reward,
                mutate_survivors=args.mutate_survivors,
                linear_policy=args.linear_policy,
                unit_flow_features=args.unit_flow_features,
                fitness_mode=args.fitness_mode,
                fitness_steps=args.fitness_steps,
                rank_replacement=args.rank_replacement,
                immigrant_fraction=args.immigrant_fraction,
                flow_head=args.flow_head,
                eval_steps=args.eval_steps,
            )
            row["proprioception"] = proprioception
            row["mutate_survivors"] = args.mutate_survivors
            row["linear_policy"] = args.linear_policy
            row["unit_flow_features"] = args.unit_flow_features
            row["fitness_mode"] = args.fitness_mode
            row["rank_replacement"] = args.rank_replacement
            row["immigrant_fraction"] = args.immigrant_fraction
            row["flow_head"] = args.flow_head
            results.append(row)
            print(
                f"seed {seed} proprio={proprioception}: "
                f"passed={row['passed']} min_delta={row['min_delta']:.3f} "
                f"min_reversed={row['min_reversed']:.3f}",
                flush=True,
            )

    passed = sum(1 for r in results if r["passed"])
    summary = {
        "status": "pilot_only",
        "seeds": args.seeds,
        "proprioception": not args.no_proprioception,
        "displacement_reward": args.displacement_reward,
        "mutate_survivors": args.mutate_survivors,
        "linear_policy": args.linear_policy,
        "unit_flow_features": args.unit_flow_features,
        "fitness_mode": args.fitness_mode,
        "fitness_steps": args.fitness_steps,
        "rank_replacement": args.rank_replacement,
        "immigrant_fraction": args.immigrant_fraction,
        "flow_head": args.flow_head,
        "generations": args.generations,
        "population": args.population,
        "steps_per_generation": args.steps_per_generation,
        "size": args.size,
        "eval_steps": args.eval_steps,
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
