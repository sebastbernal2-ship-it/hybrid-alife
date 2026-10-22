"""Run a small living world and render it to video.

This is the first vivarium layer: seasons, food, corpses, births, deaths, and
lineages, driven by the existing embodied simulation.

Usage:
    JAX_PLATFORMS=cpu python scripts/run_vivarium.py --steps 600 --size 800
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dataclasses import replace  # noqa: E402

from hybrid_alife.experiments.latent_law_embodied import embodied_law_config  # noqa: E402
from hybrid_alife.vivarium import VivariumConfig, write_video  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population", type=int, default=128)
    parser.add_argument("--world", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--size", type=int, default=800)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--season-length", type=int, default=150)
    parser.add_argument("--out", type=Path, default=Path("outputs/vivarium/vivarium.mp4"))
    parser.add_argument(
        "--poster", type=Path, default=Path("outputs/vivarium/poster.png")
    )
    args = parser.parse_args()

    sim = embodied_law_config(
        args.seed,
        generations=1,
        population=args.population,
        steps_per_generation=1,
        size=args.world,
        proprioception=True,
        initial_alive_fraction=0.6,
    )
    sim = replace(
        sim,
        world=replace(sim.world, curvature_regime="straight"),
        embodied=replace(
            sim.embodied,
            body_morphology=True,
            max_energy=80.0,
            predation_rate=0.35,
            predation_radius=0.04,
            escape_gain=0.03,
            escape_radius=0.06,
        ),
    )
    config = VivariumConfig(
        sim=sim,
        season_length=args.season_length,
        generation_length=200,
    )
    summary = write_video(
        config,
        steps=args.steps,
        size=args.size,
        fps=args.fps,
        out_path=args.out,
        warmup=args.warmup,
        poster_path=args.poster,
        seed=args.seed,
    )
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
