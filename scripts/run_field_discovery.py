#!/usr/bin/env python
"""Run the seeded latent-field discovery benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_alife.experiments.field_discovery import run_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--intervention-at", type=int, default=50)
    parser.add_argument("--indirect-noise", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None, help="JSON path; stdout if omitted")
    args = parser.parse_args()

    result = run_evaluation(
        seed=args.seed,
        repetitions=args.repetitions,
        episodes=args.episodes,
        intervention_at=args.intervention_at,
        indirect_noise=args.indirect_noise,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
        print(f"Field-discovery results written to {args.output}")


if __name__ == "__main__":
    main()
