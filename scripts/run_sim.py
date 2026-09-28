#!/usr/bin/env python
"""Run a hybrid-alife experiment from a YAML config."""

from __future__ import annotations

import argparse

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.runtime import backend_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="Requested execution backend; GPU falls back to CPU when unavailable.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    print(backend_report(args.backend))
    run_experiment(cfg)


if __name__ == "__main__":
    main()

