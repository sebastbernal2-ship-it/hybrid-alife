#!/usr/bin/env python
"""Run paired PPO-POET seeds and write machine-readable campaign artifacts."""

from __future__ import annotations

import argparse
import json
import platform
from dataclasses import asdict
from pathlib import Path

from hybrid_alife.poet import (
    POET_ARTIFACT_SCHEMA_VERSION,
    _source_commit,
    load_poet_config,
    run_poet,
)

_CONDITION_SPECS = {
    "isolated": ("isolated", "disabled"),
    "poet": ("poet", "enabled"),
    "static": ("static", "disabled"),
    "archive_disabled": ("poet", "disabled"),
    "random_archive": ("poet", "random"),
    "oracle_stepping_stone": ("poet", "oracle"),
    "avida_enabled": ("avida_enabled", "enabled"),
    "avida_persistent": ("avida_persistent", "enabled"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poet_smoke.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--out-dir", default="outputs/poet")
    parser.add_argument(
        "--tracks",
        nargs="+",
        choices=tuple(_CONDITION_SPECS),
        default=("isolated", "avida_enabled"),
    )
    parser.add_argument("--replication-id", default="replication-1")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--operator", default="unspecified")
    parser.add_argument("--machine-label", default=platform.node() or "unknown")
    parser.add_argument("--cache-cleared", action="store_true")
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="run the bounded three-seed validity pilot",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pilot:
        args.seeds = [0, 1, 2]
        args.tracks = ("poet", "avida_enabled")
        if args.generations is None:
            args.generations = 3
    root = Path(args.out_dir)
    replication_metadata = {
        "replication_id": args.replication_id,
        "seed_offset": args.seed_offset,
        "operator": args.operator,
        "machine_label": args.machine_label,
        "cache_cleared": args.cache_cleared,
    }
    manifest = {
        "schema_version": POET_ARTIFACT_SCHEMA_VERSION,
        "source_commit": _source_commit(),
        "config": args.config,
        "tracks": args.tracks,
        "seeds": args.seeds,
        "generations": args.generations,
        "replication_metadata": replication_metadata,
        "pilot": args.pilot,
        "cells": [],
    }
    root.mkdir(parents=True, exist_ok=True)
    for condition in args.tracks:
        base_track, archive_mode = _CONDITION_SPECS[condition]
        cfg = load_poet_config(
            args.config,
            track=base_track,
            archive_mode=archive_mode,
            condition=condition,
        )
        for seed in args.seeds:
            generations = args.generations if args.generations is not None else cfg.rollout_steps
            cell_dir = root / condition / f"seed-{seed:03d}"
            run_poet(
                cfg,
                seed=seed,
                generations=generations,
                out_dir=cell_dir,
                replication_metadata=replication_metadata,
            )
            manifest["cells"].append(
                {
                    "track": condition,
                    "base_track": base_track,
                    "archive_mode": archive_mode,
                    "seed": seed,
                    "generations": generations,
                    "out_dir": str(cell_dir),
                    "source_commit": manifest["source_commit"],
                    "config": asdict(cfg),
                    "replication_metadata": replication_metadata,
                    "effective_seed": seed + args.seed_offset,
                }
            )
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
