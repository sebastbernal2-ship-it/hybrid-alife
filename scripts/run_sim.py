#!/usr/bin/env python
"""Run a hybrid-alife experiment from a YAML config."""

from __future__ import annotations

import argparse

from hybrid_alife.experiments.runner import load_config, run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/base.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_experiment(cfg)


if __name__ == "__main__":
    main()

