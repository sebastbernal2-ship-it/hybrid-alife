"""Record an immersed-boundary swimmer scene into an ``HVR1`` replay for the browser viewer.

Everything drawn comes from this run: the fluid velocity (block-averaged 2x2), every
Lagrangian point of every body, and the analytic wall and post surfaces. All lengths are
lattice cells and all times lattice steps. The fluid starts at rest; no background flow.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from hybrid_alife.physics import ib
from hybrid_alife.viewer.overlap import HEAD_PAD, TAIL_ROOT, TAIL_TIP, OverlapStats
from hybrid_alife.viewer.replay import write_replay

NX, NY = 384, 224
BEAT = 2000  # steps per tail beat
FRAMES_PER_BEAT = 8
SCALE = 1.5  # linear size relative to the default ``ib.swimmer``
WAVE_AMPLITUDE = 3.0  # ib.swimmer wave_amplitude, before scaling
CONTACT_RADIUS, CONTACT_K = 3.0, 0.1
WALL_RANGE, WALL_K = 4.5, 0.1
POST = (262.0, 150.0, 17.0)  # centre x, y and radius of the solid lattice nodes
POINT_SCALE = 64.0  # int16 units per cell
VELOCITY_RANGE = 0.04  # lattice velocity at int16 full scale
# (head centre x, y, heading in degrees, wave phase in radians). Chosen so that the run
# contains a swimmer-swimmer encounter, a post encounter and a wall encounter.
SWIMMERS = (
    (90.0, 116.0, -22.0, 0.0),
    (92.0, 99.0, 22.0, 2.1),
    (228.0, 164.0, -12.0, 4.0),
    (180.0, 17.0, -28.0, 1.0),
    (340.0, 76.0, 166.0, 3.0),
)


def swimmer_body(x: float, y: float, degrees: float, phase: float) -> ib.Body:
    r = SCALE
    body = ib.swimmer(
        (x, y), np.deg2rad(degrees), head_axes=(4 * r, 2.5 * r), tail_length=20 * r,
        spacing=1.0, wave_amplitude=WAVE_AMPLITUDE, wavelength=20 * r, period=BEAT,
        # Scaling of docs/extensions/immersed_boundary.md (fixed elastoviscous number).
        stretch_k=0.05 * 2000 / (r * BEAT), bend_k=0.2 * r * 2000 / BEAT,
    )
    body.angle_phase[:] += phase * (body.angle_amp != 0)
    return body


def scene() -> tuple[ib.Structure, np.ndarray]:
    mask = np.zeros((NY, NX), np.uint8)
    mask[[0, -1]] = 1
    yy, xx = np.mgrid[:NY, :NX]
    px, py, radius = POST
    mask[(xx - px) ** 2 + (yy - py) ** 2 <= radius**2] = 1
    structure = ib.Structure.from_bodies(
        [swimmer_body(*spec) for spec in SWIMMERS],
        contact_radius=CONTACT_RADIUS, contact_k=CONTACT_K, ramp_steps=2 * BEAT,
        # Drawn surfaces sit half a cell outside the outermost solid nodes (halfway
        # bounce-back); the kernel requires points to stay 2 cells from solid nodes.
        planes=[(0.0, 1.0, 0.5), (0.0, -1.0, -(NY - 1.5))],
        discs=[(px, py, radius + 0.5)], wall_range=WALL_RANGE, wall_k=WALL_K,
    )
    return structure, mask


def _half(field: np.ndarray) -> np.ndarray:
    return field.reshape(NY // 2, 2, NX // 2, 2).mean((1, 3))


def export(
    path: str | Path, beats: int = 90, progress: Callable[[str], None] = print
) -> Path:
    """Run the scene: 2 ramp beats (not recorded), then ``beats`` recorded beats."""
    if beats < 1:
        raise ValueError("need beats >= 1")
    structure, mask = scene()
    sim = ib.ImmersedSim.create(structure, NY, NX, mask=mask).advance(2 * BEAT)
    stats = OverlapStats(structure, mask)
    stride = BEAT // FRAMES_PER_BEAT
    frames = beats * FRAMES_PER_BEAT + 1
    points = np.zeros((frames, len(structure.points), 2), np.int16)
    velocity = np.zeros((frames, NY // 2, NX // 2, 2), np.int16)
    start = [sim.centroid(i) for i in range(len(SWIMMERS))]
    for frame in range(frames):
        if frame:
            sim = sim.advance(stride)
            stats.step_report(sim.report)
        stats.frame(sim.X)
        scaled = np.rint(sim.X * POINT_SCALE)
        if np.abs(scaled).max() > 32767:
            raise ValueError("a swimmer travelled past the int16 point range")
        points[frame] = scaled
        flow = np.stack([_half(sim.u), _half(sim.v)], -1) / VELOCITY_RANGE
        if np.abs(flow).max() > 1:
            raise ValueError("flow speed exceeds VELOCITY_RANGE")
        velocity[frame] = np.rint(flow * 32767)
        if frame % (10 * FRAMES_PER_BEAT) == 0:
            progress(f"beat {frame // FRAMES_PER_BEAT}/{beats}")
    bodies, offset = [], 0
    for number in range(len(SWIMMERS)):
        count = int(np.sum(structure.body == number))
        travel = float(np.linalg.norm(sim.centroid(number) - start[number]))
        bodies.append(
            {"start": offset, "count": count, "ring": stats.rings[number], "travel": travel}
        )
        offset += count
    sample = velocity[:: max(1, frames // 40)] * (VELOCITY_RANGE / 32767)
    # derivative on the 2-cell display grid; the viewer's colour scale only
    vorticity = (np.gradient(sample[..., 1], 2.0, axis=2)
                 - np.gradient(sample[..., 0], 2.0, axis=1))
    meta = {
        "vorticity_max": float(np.percentile(np.abs(vorticity), 99.75)),
        "speed_max": float(np.percentile(np.linalg.norm(sample, axis=-1), 99.7)),
        "head_pad": HEAD_PAD, "tail_width": [TAIL_ROOT, TAIL_TIP],
        "nx": NX, "ny": NY, "nu": sim.nu, "beat_steps": BEAT, "stride": stride,
        "first_step": 2 * BEAT, "frames": frames, "point_scale": POINT_SCALE,
        "velocity_range": VELOCITY_RANGE, "bodies": bodies,
        "post": [POST[0], POST[1], POST[2] + 0.5], "walls": [0.5, NY - 1.5],
        "contact": [CONTACT_RADIUS, CONTACT_K], "wall": [WALL_RANGE, WALL_K],
        "stats": stats.summary(),
    }
    return write_replay(
        path, meta, {"points": (points, "delta-shuffle"), "velocity": (velocity, "delta-shuffle")}
    )
