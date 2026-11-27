"""Grid-refinement studies for the immersed-boundary free swimmer and Taylor sheet.

Refinement factor r multiplies every length (cells), and the beat period by r^2, at fixed
lattice viscosity (so the wave Reynolds number is fixed). Point spacing stays a fixed number
of cells. In 2-D, per unit depth, a force then scales as mu U ~ 1/r and a bending rigidity
B = force * length^2 as r. With a fixed point spacing in cells the joint stiffness B/ds
therefore scales as r and the spring constant EA/ds as 1/r ("consistent"). "as_written" is
the formula in the earlier text of docs/extensions/immersed_boundary.md read literally
(bend_k ~ r/period, stretch_k ~ 1/(r period) with the period in steps), which is softer by
1/r^2. It does not reproduce the old table; "consistent" does.

Each case is one JSON file in the output directory; existing files are not recomputed.
    python scripts/run_ib_convergence.py --study swimmer --jobs 6
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from hybrid_alife.physics import ib

NU = 1 / 6
SAMPLES = 40  # per beat


def swimmer_case(r: float, scaling: str, beats: int = 6, box: float = 1.0,
                 period0: float = 2000.0, stiff: float = 1.0) -> dict:
    period = int(round(period0 * r * r / SAMPLES)) * SAMPLES
    speed_factor = 2000.0 / period0  # forces are proportional to the beat frequency
    if scaling == "consistent":
        bend, stretch = 0.2 * r, 0.05 / r
    elif scaling == "as_written":
        bend, stretch = 0.2 / r, 0.05 / r**3
    else:
        raise ValueError(scaling)
    bend, stretch = bend * speed_factor * stiff, stretch * speed_factor * stiff
    ny, nx = int(round(64 * r * box)), int(round(96 * r * box))
    body = ib.swimmer(
        (40.0 * r, ny / 2 + 0.21), head_axes=(4.0 * r, 2.5 * r), tail_length=20.0 * r,
        wave_amplitude=1.5 * r, wavelength=20.0 * r, period=period,
        stretch_k=stretch, bend_k=bend)
    s = ib.Structure.from_bodies([body], ramp_steps=period)
    sim = ib.ImmersedSim.create(s, ny, nx, nu=NU)
    tip = len(s.points) - 1
    rear = 1 + (tip - int(round(20.0 * r)) - 1) // 2 + 1
    strides, headings, tips, strains, speeds, drho = [], [], [], [], [], []
    for _ in range(beats):
        x0 = sim.centroid()
        lateral, angle = [], []
        for _ in range(SAMPLES):
            sim = sim.advance(period // SAMPLES)
            axis = sim.X[0] - sim.X[rear]  # tail root to hub: the head axis
            axis /= np.linalg.norm(axis)
            d = sim.X[tip] - sim.X[0]
            lateral.append((axis[0] * d[1] - axis[1] * d[0]) / r)
            angle.append(np.arctan2(axis[1], axis[0]))
            length = np.linalg.norm(sim.X[s.edges[:, 1]] - sim.X[s.edges[:, 0]], axis=1)
            strains.append(float(np.max(np.abs(length / s.edge_rest - 1))))
            speeds.append(sim.report["max_speed"])
            drho.append(float(np.ptp(sim.rho)))
        dx = (sim.centroid() - x0) / r
        strides.append(dx.tolist())
        headings.append(float(np.mean(angle)))
        phase = np.exp(-2j * np.pi * np.arange(SAMPLES) / SAMPLES)
        tips.append(float(2 * abs(np.mean(np.asarray(lateral) * phase))))
    return {"study": "swimmer", "r": r, "scaling": scaling, "beats": beats, "box": box,
            "period": period, "period0": period0, "stiff": stiff, "bend_k": bend,
            "stretch_k": stretch, "ny": ny, "nx": nx, "n_points": len(s.points),
            "stride_xy_per_beat": strides, "heading_per_beat": headings,
            "tip_amplitude_per_beat": tips, "max_strain": max(strains),
            "max_speed": max(speeds), "max_density_range": max(drho)}


def sheet_case(n: int, height: int = 2, scaling: str = "consistent", spacing: float = 0.5,
               beats: int = 3, stiff: float = 1.0) -> dict:
    """Taylor sheet, `n` cells per wavelength, box `height` wavelengths tall."""
    r = n / 48
    period = int(round(9000 * r * r / SAMPLES)) * SAMPLES
    if scaling != "consistent":
        raise ValueError(scaling)
    bend, stretch = 3.0 * r * stiff, 0.1 / r * stiff
    ny, k, omega = n * height, 2 * np.pi / n, 2 * np.pi / period
    body = ib.sheet(n, ny / 2 + 0.17, 0.2 / k, period=period, spacing=spacing,
                    bend_k=bend * 0.5 / spacing, stretch_k=stretch * 0.5 / spacing)
    s = ib.Structure.from_bodies([body], ramp_steps=period)
    sim = ib.ImmersedSim.create(s, ny, n, nu=NU)
    label = s.points[:, 0]  # material coordinate
    speeds = []
    modes = []
    for beat in range(beats):
        x0, far = sim.X[:, 0].mean(), []
        for j in range(SAMPLES):
            sim = sim.advance(period // SAMPLES)
            far.append(sim.u[int(round(sim.X[:, 1].mean() + ny / 2)) % ny].mean())
            if beat == beats - 1:
                wave = np.exp(-1j * k * label)
                x = sim.X[:, 0] - label
                y = sim.X[:, 1] - sim.X[:, 1].mean()
                modes.append((2 * np.mean((x - x.mean()) * wave), 2 * np.mean(y * wave),
                              2 * abs(np.mean(y * np.exp(-1j * k * sim.X[:, 0])))))
        speeds.append(float((sim.X[:, 0].mean() - x0) / period - np.mean(far)))
    ax, by, amp = (np.array(c) for c in zip(*modes, strict=True))
    amp = float(np.mean(amp.real))
    tuck = np.sqrt((1 + np.sqrt(1 + (omega / (NU * k**2)) ** 2)) / 2)
    tuck = (1 + tuck) / (2 * tuck)
    theory = 0.5 * omega * k * amp**2 * (1 - 19 / 16 * (amp * k) ** 2) * tuck
    return {"study": "sheet", "n": n, "height": height, "scaling": scaling,
            "spacing": spacing, "beats": beats, "stiff": stiff, "period": period,
            "bend_k": bend, "stretch_k": stretch, "speed_per_beat": speeds,
            "amplitude": amp, "amplitude_over_prescribed": amp * k / 0.2,
            "longitudinal_over_transverse": float(np.mean(abs(ax)) / np.mean(abs(by))),
            "theory": float(theory), "error": float(-speeds[-1] / theory - 1)}


def cases(study: str) -> dict:
    out = {}
    if study == "swimmer":
        for scaling in ("as_written", "consistent"):
            for r in (1, 1.5, 2, 3, 4):
                if scaling == "as_written" and r > 2:
                    continue
                out[f"swimmer_{scaling}_r{r}"] = (swimmer_case, dict(r=r, scaling=scaling))
        out["swimmer_consistent_r6"] = (
            swimmer_case, dict(r=6, scaling="consistent", beats=4))
    elif study == "sheet":
        for n in (32, 48, 64, 96, 128, 192):
            for h in (1, 2):
                out[f"sheet_n{n}_h{h}"] = (sheet_case, dict(n=n, height=h))
    return out


def run(name: str, fn, kwargs: dict, out: Path) -> str:
    result = fn(**kwargs)
    (out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
    return name


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--study", required=True)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--only", nargs="*", help="case names")
    p.add_argument("--output", type=Path, default=Path("docs/results/ib_convergence/raw"))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    todo = {k: v for k, v in cases(a.study).items()
            if (not a.only or k in a.only) and not (a.output / f"{k}.json").exists()}
    with ProcessPoolExecutor(a.jobs) as pool:
        for done in as_completed([pool.submit(run, k, *v, a.output) for k, v in todo.items()]):
            print("done", done.result(), flush=True)


if __name__ == "__main__":
    main()
