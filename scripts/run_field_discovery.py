#!/usr/bin/env python
"""Run the seeded schema-v4 hybrid-alife field-discovery pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_alife.experiments.field_discovery import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--training-generations", type=int, default=2)
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON path; stdout if omitted")
    args = parser.parse_args()
    result = run_evaluation(
        seed=args.seed,
        steps=args.steps,
        training_generations=args.training_generations,
        config_dir=args.config_dir,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Field-discovery pilot written to {args.output}")


if __name__ == "__main__":
    main()
