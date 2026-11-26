"""Opt-in smoke: two swimmers with contact, state exported for a viewer.

    python scripts/run_ib_smoke.py --output outputs/ib_smoke.jsonl

Standalone: it does not touch the world simulator or evolution.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hybrid_alife.physics import ib
from hybrid_alife.physics.units import UnitMap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/ib_smoke.jsonl")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--steps-per-frame", type=int, default=200)
    parser.add_argument("--dx-um", type=float, default=0.25)
    args = parser.parse_args()
    units = UnitMap(dx_m=args.dx_um * 1e-6)
    structure = ib.Structure.from_bodies(
        [ib.swimmer((40.0, 36.0)), ib.swimmer((88.0, 44.0), heading=3.14159)],
        contact_radius=1.5, contact_k=0.02, ramp_steps=2000)
    start = ib.ImmersedSim.create(structure, 80, 128)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    end = ib.record_states(start, args.output, args.frames, args.steps_per_frame,
                           dx_m=units.dx_m, dt_s=units.dt_s)
    for body in range(len(structure.kinds)):
        moved = end.centroid(body) - start.centroid(body)
        print(f"body {body}: moved {moved.round(3).tolist()} cells in {end.step} steps")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
