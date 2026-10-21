"""Run the D1 flow-head ablation.

Each condition changes one mechanism relative to the full configuration. The
driver runs the existing study script for every condition and aggregates the
per-condition summaries.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_d1_ablation.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

STUDY = Path(__file__).with_name("run_embodied_law_study.py")

# Flags that define the full, accepted configuration.
FULL_FLAGS = [
    "--unit-flow-features",
    "--flow-head",
    "--fitness-mode",
    "min_alignment",
    "--rank-replacement",
    "--immigrant-fraction",
    "0.25",
]

# One mechanism removed per condition. The list is intentionally explicit so a
# typo cannot silently change two mechanisms at once.
CONDITIONS: dict[str, list[str]] = {
    "full": list(FULL_FLAGS),
    "no_flow_head": [
        "--unit-flow-features",
        "--linear-policy",
        "--fitness-mode",
        "min_alignment",
        "--rank-replacement",
        "--immigrant-fraction",
        "0.25",
    ],
    "raw_flow": [
        "--raw-flow-features",
        "--flow-head",
        "--fitness-mode",
        "min_alignment",
        "--rank-replacement",
        "--immigrant-fraction",
        "0.25",
    ],
    "energy_fitness": [
        "--unit-flow-features",
        "--flow-head",
        "--rank-replacement",
        "--immigrant-fraction",
        "0.25",
    ],
    "no_rank": [
        "--unit-flow-features",
        "--flow-head",
        "--fitness-mode",
        "min_alignment",
        "--immigrant-fraction",
        "0.0",
    ],
    "no_immigrants": [
        "--unit-flow-features",
        "--flow-head",
        "--fitness-mode",
        "min_alignment",
        "--rank-replacement",
        "--immigrant-fraction",
        "0.0",
    ],
}


def build_command(condition: str, *, seeds: list[int], generations: int, output: Path) -> list[str]:
    if condition not in CONDITIONS:
        raise KeyError(f"unknown condition: {condition}")
    return [
        sys.executable,
        str(STUDY),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--generations",
        str(generations),
        "--population",
        "16",
        "--steps-per-generation",
        "8",
        "--size",
        "14",
        "--eval-steps",
        "12",
        "--fitness-steps",
        "6",
        "--displacement-reward",
        "5.0",
        *CONDITIONS[condition],
        "--output",
        str(output),
    ]


def summarize(summary: dict) -> dict:
    results = summary["results"]
    return {
        "seeds_passed": summary["seeds_passed"],
        "seeds_total": summary["seeds_total"],
        "worst_delta": min(r["min_delta"] for r in results),
        "worst_evolved": min(r["min_evolved"] for r in results),
        "worst_reversed": min(r["min_reversed"] for r in results),
        "per_seed": [
            {
                "seed": r["seed"],
                "passed": r["passed"],
                "min_delta": r["min_delta"],
                "min_evolved": r["min_evolved"],
                "min_reversed": r["min_reversed"],
            }
            for r in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("docs/results/d1_ablation"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    aggregate: dict[str, dict] = {}
    for condition in CONDITIONS:
        summary_path = args.output / f"{condition}.json"
        print(f"\n=== {condition} ===", flush=True)
        completed = subprocess.run(
            build_command(
                condition,
                seeds=args.seeds,
                generations=args.generations,
                output=summary_path,
            )
        )
        if summary_path.exists():
            aggregate[condition] = summarize(json.loads(summary_path.read_text()))
        else:
            aggregate[condition] = {"error": f"exit {completed.returncode}"}

    (args.output / "ablation.json").write_text(
        json.dumps(
            {
                "generations": args.generations,
                "seeds": args.seeds,
                "conditions": aggregate,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print("\ncondition        seeds  worst_delta  worst_evolved  worst_reversed")
    for condition, row in aggregate.items():
        if "error" in row:
            print(f"{condition:<16} ERROR {row['error']}")
            continue
        print(
            f"{condition:<16} {row['seeds_passed']}/{row['seeds_total']}   "
            f"{row['worst_delta']:+.3f}      {row['worst_evolved']:+.3f}        "
            f"{row['worst_reversed']:+.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
