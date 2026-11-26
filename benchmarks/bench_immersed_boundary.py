"""Single-core throughput of the immersed-boundary step. `python benchmarks/<this file>`."""

from __future__ import annotations

import platform
import time

import numpy as np

from hybrid_alife.physics import ib, lbm


def fluid_only(ny: int, nx: int, steps: int) -> None:
    """The existing live solver with no bodies, as the baseline."""
    fluid = lbm.LiveLBM.uniform(nx, ny)  # LiveLBM's world frame is the transpose
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        fluid.advance(steps)
        best = min(best, time.perf_counter() - start)
    print(f"{ny}x{nx}  live solver, no bodies{'':28s}{steps / best:9.0f} steps/s  "
          f"{ny * nx * steps / best / 1e6:6.1f} MLUPS", flush=True)


def case(ny: int, nx: int, n_bodies: int, contact: bool, steps: int) -> None:
    columns = int(np.ceil(np.sqrt(n_bodies)))
    centres = [
        ((i % columns + 0.5) * nx / columns, (i // columns + 0.5) * ny / columns)
        for i in range(n_bodies)
    ]
    bodies = [ib.swimmer(c) for c in centres]
    structure = ib.Structure.from_bodies(
        bodies, ramp_steps=1000,
        contact_radius=1.5 if contact else 0.0, contact_k=0.02 if contact else 0.0)
    sim = ib.ImmersedSim.create(structure, ny, nx).advance(20)
    best = np.inf
    for _ in range(3):
        start = time.perf_counter()
        sim.advance(steps)
        best = min(best, time.perf_counter() - start)
    points = len(structure.points)
    print(f"{ny}x{nx}  bodies={n_bodies:3d}  points={points:5d}  contact={int(contact)}  "
          f"{steps / best:9.0f} steps/s  {ny * nx * steps / best / 1e6:6.1f} MLUPS", flush=True)


if __name__ == "__main__":
    print(platform.processor() or platform.machine(), "(single thread, best of 3)")
    fluid_only(64, 96, 2000)
    case(64, 96, 1, False, 2000)
    case(128, 128, 4, True, 600)
    fluid_only(256, 256, 200)
    case(256, 256, 16, False, 200)
    case(256, 256, 16, True, 200)
    case(256, 256, 64, True, 200)
