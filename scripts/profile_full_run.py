#!/usr/bin/env python
"""Record a full-run CPU profile before adopting optimizations."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from io import StringIO
from pathlib import Path

from hybrid_alife.poet import load_poet_config, run_poet
from hybrid_alife.runtime import backend_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/poet_smoke.yaml")
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg = load_poet_config(args.config, track="isolated")
    profiler = cProfile.Profile()
    started = time.perf_counter()
    profiler.enable()
    run_poet(cfg, seed=args.seed, generations=args.generations)
    profiler.disable()
    elapsed = time.perf_counter() - started
    stream = StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats("cumulative").print_stats(30)
    payload = {
        "config": args.config,
        "seed": args.seed,
        "generations": args.generations,
        "backend": backend_report("auto"),
        "wall_seconds": elapsed,
        "optimization_adoption": "world_and_avida_step_jit_adopted_after_preoptimization_profile",
        "profile": stream.getvalue(),
    }
    Path(args.out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in payload.items() if k != "profile"}, indent=2))


if __name__ == "__main__":
    main()
