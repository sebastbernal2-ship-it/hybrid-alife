#!/usr/bin/env python
"""Run the deterministic selective-causal-inquiry calibration benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hybrid_alife.experiments.awareness_benchmark import evaluate_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    artifact = evaluate_benchmark(tuple(args.seeds), horizon=args.horizon)
    payload = json.dumps(artifact, indent=2) + "\n"
    if args.output is None:
        print(payload, end="")
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(f"Awareness benchmark written to {args.output}")


if __name__ == "__main__":
    main()
