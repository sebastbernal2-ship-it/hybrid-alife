#!/usr/bin/env python
"""Run paired PPO-POET seeds and write machine-readable campaign artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from hybrid_alife.poet import _source_commit, load_poet_config, run_poet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poet_smoke.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/poet")
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=("isolated", "avida_enabled", "avida_persistent"),
        default=("isolated", "avida_enabled"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out_dir)
    manifest = {
        "source_commit": _source_commit(),
        "config": args.config,
        "tracks": args.tracks,
        "seeds": args.seeds,
        "generations": args.generations,
        "cells": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    for track in args.tracks:
        cfg = load_poet_config(args.config, track=track)
        for seed in args.seeds:
            generations = args.generations if args.generations is not None else cfg.rollout_steps
            cell_dir = root / track / f"seed-{seed:03d}"
            run_poet(cfg, seed=seed, generations=generations, out_dir=cell_dir)
            manifest["cells"].append(
                {
                    "track": track,
                    "seed": seed,
                    "generations": generations,
                    "out_dir": str(cell_dir),
                    "source_commit": manifest["source_commit"],
                    "config": asdict(cfg),
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
