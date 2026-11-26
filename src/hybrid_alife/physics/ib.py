"""Opt-in immersed-boundary bodies on the D2Q9 live solver (native/ib.cpp, contact.cpp).

Everything here is in the LBM frame and lattice units: grids are [ny, nx], points are
(x, y) with node (i, j) at (x=i, y=j), one time step is 1. Points are unwrapped (never
folded into the box) so that displacements can be read off directly. Nothing in the
simulator or in evolution uses this module. See docs/extensions/immersed_boundary.md.
"""

from __future__ import annotations

import ctypes
import json
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from hybrid_alife.physics import lbm

_DIR = Path(__file__).parent
_NATIVE_DIR = Path(__file__).resolve().parents[3] / "native"
_libs: tuple[ctypes.CDLL, ctypes.CDLL] | None = None
STATE_SCHEMA = "hybrid-alife/ib-state/v1"

_ERRORS = {
    -1: "invalid arguments (need ny, nx >= 4, nu > 0, valid indices, finite points)",
    -2: "a body came within the delta-kernel support (2 cells) of a solid node",
    -3: "unstable: non-finite or degenerate state, or speed above 0.3 lattice units; "
    "lower the spring stiffness or the actuation, or raise the viscosity",
}


class _Structure(ctypes.Structure):
    _fields_ = [
        ("n_points", ctypes.c_int32),
        ("n_edges", ctypes.c_int32),
        ("edges", ctypes.c_void_p),
        ("edge_rest", ctypes.c_void_p),
        ("edge_k", ctypes.c_void_p),
        ("edge_shift", ctypes.c_void_p),
        ("n_angles", ctypes.c_int32),
        ("angles", ctypes.c_void_p),
        ("angle_base", ctypes.c_void_p),
        ("angle_amp", ctypes.c_void_p),
        ("angle_phase", ctypes.c_void_p),
        ("angle_omega", ctypes.c_void_p),
        ("angle_k", ctypes.c_void_p),
        ("angle_shift", ctypes.c_void_p),
        ("n_tethers", ctypes.c_int32),
        ("tethers", ctypes.c_void_p),
        ("tether_target", ctypes.c_void_p),
        ("tether_k", ctypes.c_double),
        ("body", ctypes.c_void_p),
        ("contact_radius", ctypes.c_double),
        ("contact_k", ctypes.c_double),
        ("ramp_steps", ctypes.c_double),
    ]


class _Report(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_double)
        for name in (
            "max_net_force",
            "max_net_torque",
            "max_force_scale",
            "tether_fx",
            "tether_fy",
            "min_contact_distance",
            "max_speed",
        )
    ]


def _load() -> tuple[ctypes.CDLL, ctypes.CDLL]:
    """(libib, libcontact), building the native kernels on first use."""
    global _libs
    if _libs is None:
        paths = (_DIR / "libib.so", _DIR / "libcontact.so")
        if not all(p.exists() for p in paths):
            try:
                subprocess.run(
                    ["make", "-C", str(_NATIVE_DIR)], check=True, capture_output=True, text=True
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(
                    "building the native immersed-boundary kernels failed (needs g++ and make; "
                    f"run `make -C native`):\n{detail}"
                ) from exc
        ib, contact = (ctypes.CDLL(str(p)) for p in paths)
        vp, dbl, i = ctypes.c_void_p, ctypes.c_double, ctypes.c_int
        sp = ctypes.POINTER(_Structure)
        ib.ib_forces.argtypes = [sp, vp, dbl, vp]
        ib.ib_forces.restype = dbl
        ib.ib_spread.argtypes = [i, i, i, vp, vp, vp]
        ib.ib_spread.restype = None
        ib.ib_advance.argtypes = [
            i, i, vp, dbl, dbl, dbl, sp, i, dbl, vp, vp, vp, vp, vp, vp,
            ctypes.POINTER(_Report),
        ]
        ib.ib_advance.restype = i
        contact.contact_forces.argtypes = [i, vp, vp, dbl, dbl, dbl, dbl, i, vp, vp]
        contact.contact_forces.restype = ctypes.c_long
        _libs = ib, contact
    return _libs


def _array(value, dtype, shape) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(value, dtype=dtype).reshape(shape))


@dataclass(frozen=True)
class Body:
    """One body: points plus the springs that give it shape. Indices are body-local.

    A joint (i, j, k) has signed angle 0 when straight; its rest angle at time t is
    `angle_base + ramp(t) * angle_amp * sin(angle_omega * t - angle_phase)`.
    `periodic` is the (x, y) box period a filament closes across (used by the sheet).
    """

    points: np.ndarray
    kind: str
    edges: np.ndarray = ()
    edge_k: np.ndarray = ()
    angles: np.ndarray = ()
    angle_base: np.ndarray = ()
    angle_amp: np.ndarray = ()
    angle_phase: np.ndarray = ()
    angle_omega: np.ndarray = ()
    angle_k: np.ndarray = ()
    tethered: bool = False
    periodic: tuple[float, float] = (0.0, 0.0)


def cylinder(center: tuple[float, float], radius: float, spacing: float = 0.75) -> Body:
    """Rigid cylinder: a ring of points, each tethered to where it started."""
    count = max(8, int(np.ceil(2 * np.pi * radius / spacing)))
    a = 2 * np.pi * np.arange(count) / count
    pts = np.asarray(center) + radius * np.stack([np.cos(a), np.sin(a)], 1)
    return Body(pts, "cylinder", tethered=True)


def swimmer(
    center: tuple[float, float],
    heading: float = 0.0,
    head_axes: tuple[float, float] = (4.0, 2.5),
    tail_length: float = 20.0,
    spacing: float = 1.0,
    wave_amplitude: float = 1.5,
    wavelength: float = 20.0,
    period: float = 2000.0,
    stretch_k: float = 0.05,
    bend_k: float = 0.2,
    standing: bool = False,
) -> Body:
    """Elliptical cell body plus a flexible tail driven by a travelling curvature wave.

    The head is a hub-and-ring truss (rigid when springs are stiff). The tail's rest
    curvature is `wave_amplitude * k^2 * sin(omega t - k s)`: a wave running from head to
    tip, so the body swims along `heading`. `standing=True` replaces it by a standing wave
    (reciprocal actuation, which cannot give net motion at zero Reynolds number).
    """
    a, b = head_axes
    ring = max(8, 2 * int(np.ceil(np.pi * (a + b) / 2 / spacing)))
    phi = 2 * np.pi * np.arange(ring) / ring
    pts = [(0.0, 0.0)] + [(a * np.cos(p), b * np.sin(p)) for p in phi]
    rear = 1 + ring // 2
    n_tail = max(2, int(round(tail_length / spacing)))
    ds = tail_length / n_tail
    chain = [rear]
    for m in range(1, n_tail + 1):
        pts.append((-a - m * ds, 0.0))
        chain.append(len(pts) - 1)
    edges = [(0, 1 + m) for m in range(ring)]
    edges += [(1 + m, 1 + (m + 1) % ring) for m in range(ring)]
    edges += list(zip(chain[:-1], chain[1:], strict=True))
    joints = [(0, rear, chain[1])]
    joints += [(chain[m - 1], chain[m], chain[m + 1]) for m in range(1, n_tail)]
    k = 2 * np.pi / wavelength
    s = np.arange(len(joints)) * ds
    amp = wave_amplitude * k**2 * ds * np.ones(len(joints))
    amp[0] = 0.0  # the root stays straight and anchors the tail to the head
    phase = k * s
    if standing:
        amp, phase = amp * np.cos(phase), np.zeros_like(phase)
    c, sn = np.cos(heading), np.sin(heading)
    xy = np.asarray(pts) @ np.array([[c, sn], [-sn, c]]) + np.asarray(center)
    return Body(
        xy, "swimmer", np.asarray(edges), np.full(len(edges), stretch_k),
        np.asarray(joints), np.zeros(len(joints)), amp, phase,
        np.full(len(joints), 2 * np.pi / period), np.full(len(joints), bend_k),
    )


def sheet(
    nx: int,
    y: float,
    amplitude: float,
    waves: int = 1,
    period: float = 4000.0,
    spacing: float = 0.5,
    stretch_k: float = 0.1,
    bend_k: float = 1.0,
) -> Body:
    """Taylor's swimming sheet: a periodic filament across the box, wave running toward +x."""
    count = int(round(nx / spacing))
    ds = nx / count
    x = ds * np.arange(count)
    k = 2 * np.pi * waves / nx
    idx = np.arange(count)
    edges = np.stack([idx, (idx + 1) % count], 1)
    joints = np.stack([(idx - 1) % count, idx, (idx + 1) % count], 1)
    return Body(
        np.stack([x, np.full(count, float(y))], 1), "sheet", edges, np.full(count, stretch_k),
        joints, np.zeros(count), np.full(count, amplitude * k**2 * ds), k * x,
        np.full(count, 2 * np.pi / period), np.full(count, bend_k), periodic=(float(nx), 0.0),
    )


@dataclass(frozen=True)
class Structure:
    """Several bodies flattened into the arrays the kernel reads."""

    points: np.ndarray  # [N, 2] initial positions
    body: np.ndarray
    kinds: tuple[str, ...]
    edges: np.ndarray
    edge_rest: np.ndarray
    edge_k: np.ndarray
    edge_shift: np.ndarray
    angles: np.ndarray
    angle_base: np.ndarray
    angle_amp: np.ndarray
    angle_phase: np.ndarray
    angle_omega: np.ndarray
    angle_k: np.ndarray
    angle_shift: np.ndarray
    tethers: np.ndarray
    tether_target: np.ndarray
    tether_k: float = 0.0
    contact_radius: float = 0.0
    contact_k: float = 0.0
    ramp_steps: float = 0.0

    @classmethod
    def from_bodies(
        cls,
        bodies: Sequence[Body],
        tether_k: float = 0.02,
        contact_radius: float = 0.0,
        contact_k: float = 0.0,
        ramp_steps: float = 0.0,
    ) -> Structure:
        if not bodies:
            raise ValueError("a structure needs at least one body")
        cols: dict[str, list] = {k: [] for k in (
            "points", "body", "edges", "edge_rest", "edge_k", "edge_shift", "angles",
            "angle_base", "angle_amp", "angle_phase", "angle_omega", "angle_k",
            "angle_shift", "tethers", "tether_target")}
        offset = 0
        for number, b in enumerate(bodies):
            pts = _array(b.points, np.float64, (-1, 2))
            n = len(pts)
            period = np.asarray(b.periodic, np.float64)

            def shifts(vec, period=period):
                """Periodic offset that brings a separation to its nearest image."""
                out = np.zeros_like(vec)
                on = period > 0
                out[:, on] = -period[on] * np.rint(vec[:, on] / period[on])
                return out

            edges = _array(b.edges, np.int32, (-1, 2))
            joints = _array(b.angles, np.int32, (-1, 3))
            for name, idx in (("edges", edges), ("angles", joints)):
                if idx.size and not (0 <= idx.min() and idx.max() < n):
                    raise ValueError(f"body {number}: {name} index out of range")
            d = pts[edges[:, 1]] - pts[edges[:, 0]]
            es = shifts(d)
            cols["points"].append(pts)
            cols["body"].append(np.full(n, number, np.int32))
            cols["edges"].append(edges + offset)
            cols["edge_rest"].append(np.linalg.norm(d + es, axis=1))
            cols["edge_k"].append(_array(b.edge_k, np.float64, len(edges)))
            cols["edge_shift"].append(es)
            cols["angles"].append(joints + offset)
            for name in ("angle_base", "angle_amp", "angle_phase", "angle_omega", "angle_k"):
                cols[name].append(_array(getattr(b, name), np.float64, len(joints)))
            cols["angle_shift"].append(np.concatenate(
                [shifts(pts[joints[:, 0]] - pts[joints[:, 1]]),
                 shifts(pts[joints[:, 2]] - pts[joints[:, 1]])], 1))
            if b.tethered:
                cols["tethers"].append(np.arange(offset, offset + n, dtype=np.int32))
                cols["tether_target"].append(pts)
            offset += n
        empty = {"tethers": np.zeros(0, np.int32), "tether_target": np.zeros((0, 2))}
        joined = {
            k: np.ascontiguousarray(np.concatenate(v)) if v else empty[k] for k, v in cols.items()
        }
        return cls(kinds=tuple(b.kind for b in bodies), tether_k=tether_k,
                   contact_radius=contact_radius, contact_k=contact_k, ramp_steps=ramp_steps,
                   **joined)

    def _c(self) -> _Structure:
        # The arrays are owned by self, which outlives every call that uses the pointers.
        def p(name):
            return getattr(self, name).ctypes.data

        names = [f[0] for f in _Structure._fields_]
        values = {k: p(k) for k in names if isinstance(getattr(self, k, None), np.ndarray)}
        return _Structure(
            n_points=len(self.points), n_edges=len(self.edges), n_angles=len(self.angles),
            n_tethers=len(self.tethers), tether_k=self.tether_k,
            contact_radius=self.contact_radius, contact_k=self.contact_k,
            ramp_steps=self.ramp_steps, **values,
        )

    def forces(self, X: np.ndarray, t: float) -> tuple[float, np.ndarray]:
        """Elastic energy and internal forces at time t (no contact, no tethers)."""
        X = _array(X, np.float64, (len(self.points), 2))
        F = np.zeros_like(X)
        energy = _load()[0].ib_forces(ctypes.byref(self._c()), X.ctypes.data, t, F.ctypes.data)
        if not np.isfinite(energy):
            raise ValueError("invalid or degenerate structure")
        return energy, F


def spread(shape: tuple[int, int], X: np.ndarray, F: np.ndarray) -> np.ndarray:
    """Spread point forces onto a [ny, nx, 2] force-density grid with the 4-point delta."""
    X = _array(X, np.float64, (-1, 2))
    F = _array(F, np.float64, X.shape)
    if not np.all(np.abs(X) < 1e8):  # the kernel casts positions to int
        raise ValueError("points must be finite and below 1e8 in magnitude")
    grid = np.zeros((*shape, 2))
    _load()[0].ib_spread(
        shape[0], shape[1], len(X), X.ctypes.data, F.ctypes.data, grid.ctypes.data
    )
    return grid


def contact_forces(
    X: np.ndarray, body: np.ndarray, box: tuple[float, float], radius: float, k: float,
    brute: bool = False,
) -> tuple[np.ndarray, int, float]:
    """(forces [N, 2], contacting pair count, min distance) in a periodic (Lx, Ly) box."""
    X = _array(X, np.float64, (-1, 2))
    body = _array(body, np.int32, len(X))
    F = np.zeros_like(X)
    dmin = ctypes.c_double(radius)
    pairs = _load()[1].contact_forces(
        len(X), X.ctypes.data, body.ctypes.data, box[0], box[1], radius, k, int(brute),
        F.ctypes.data, ctypes.byref(dmin),
    )
    if pairs < 0:
        raise ValueError("invalid contact arguments (need 0 < 2*radius < box, finite points)")
    return F, int(pairs), dmin.value


@dataclass(frozen=True)
class ImmersedSim:
    """Fluid and bodies. `advance` returns a new object and never mutates this one."""

    structure: Structure
    f: np.ndarray  # [ny, nx, 9], same layout as LiveLBM.f
    mask: np.ndarray
    nu: float
    X: np.ndarray  # [N, 2] unwrapped point positions
    U: np.ndarray  # [N, 2] point velocities of the last step
    u: np.ndarray
    v: np.ndarray
    rho: np.ndarray
    body_force: tuple[float, float] = (0.0, 0.0)
    step: int = 0
    report: dict | None = None  # diagnostics of the last advance() call

    @classmethod
    def create(
        cls,
        structure: Structure,
        ny: int,
        nx: int,
        nu: float = 1 / 6,
        mask: np.ndarray | None = None,
        body_force: tuple[float, float] = (0.0, 0.0),
    ) -> ImmersedSim:
        """Fluid at rest with density 1. `mask` marks solid nodes (e.g. channel walls)."""
        mask = np.zeros((ny, nx), np.uint8) if mask is None else np.asarray(mask) != 0
        mask = np.ascontiguousarray(mask, np.uint8)
        if mask.shape != (ny, nx):
            raise ValueError(f"mask must have shape {(ny, nx)}, got {mask.shape}")
        zero = np.zeros((ny, nx))
        f = np.zeros((ny, nx, 9))
        lbm._load().lbm_init(ny, nx, 1.0, zero, zero.copy(), f)
        X = np.array(structure.points, np.float64)
        return cls(structure, f, mask, nu, X, np.zeros_like(X), zero, zero, np.ones((ny, nx)),
                   body_force)

    @classmethod
    def from_live(cls, structure: Structure, fluid: lbm.LiveLBM) -> ImmersedSim:
        """Put bodies into a copy of a LiveLBM fluid (LBM frame: x = world axis 0).

        The live fluid's constant body force is kept; its mean-flow relaxation is not.
        """
        X = np.array(structure.points, np.float64)
        return cls(structure, np.ascontiguousarray(fluid.f.copy()), fluid.mask.copy(), fluid.nu,
                   X, np.zeros_like(X), fluid.u.copy(), fluid.v.copy(), fluid.rho.copy(),
                   (fluid.body_force, 0.0))

    def advance(self, steps: int) -> ImmersedSim:
        ny, nx = self.mask.shape
        points = (len(self.structure.points), 2)
        for name, shape in (("f", (ny, nx, 9)), ("X", points), ("U", points)):
            if np.shape(getattr(self, name)) != shape:  # the kernel would overrun it
                raise ValueError(f"{name} must have shape {shape}")
        if self.mask.dtype != np.uint8 or not self.mask.flags.c_contiguous:
            raise ValueError("mask must be a C-contiguous uint8 array")
        f, X, U = (np.array(a, np.float64, order="C") for a in (self.f, self.X, self.U))
        u, v, rho = (np.zeros((ny, nx)) for _ in range(3))
        report = _Report()
        rc = _load()[0].ib_advance(
            ny, nx, self.mask.ctypes.data, self.nu, *self.body_force,
            ctypes.byref(self.structure._c()), steps, self.step, f.ctypes.data,
            X.ctypes.data, U.ctypes.data, u.ctypes.data, v.ctypes.data, rho.ctypes.data,
            ctypes.byref(report),
        )
        if rc < 0:
            raise ValueError(f"immersed-boundary step failed: {_ERRORS.get(rc, rc)}")
        return replace(
            self, f=f, X=X, U=U, u=u, v=v, rho=rho, step=self.step + steps,
            report={name: getattr(report, name) for name, _ in _Report._fields_},
        )

    def fluid_momentum(self) -> np.ndarray:
        """Raw momentum sum(f c) over fluid nodes, (x, y)."""
        fluid = (self.mask == 0)[..., None]
        return np.array([np.sum(self.f * lbm._CX * fluid), np.sum(self.f * lbm._CY * fluid)])

    def centroid(self, body: int = 0) -> np.ndarray:
        return self.X[self.structure.body == body].mean(0)


def record_states(
    sim: ImmersedSim, path: str | Path, frames: int, steps_per_frame: int,
    dx_m: float | None = None, dt_s: float | None = None,
) -> ImmersedSim:
    """Advance `sim`, writing JSON Lines: one header, then `frames + 1` point frames.

    Header: schema, nx, ny, dx_m, dt_s (null if no physical mapping), and per body its
    kind, [start, stop) point range and body-local edges. Frame: step and points [[x, y]]
    in lattice units, unwrapped (wrap modulo nx, ny to draw).
    """
    s = sim.structure
    bodies = []
    for number, kind in enumerate(s.kinds):
        idx = np.flatnonzero(s.body == number)
        start, stop = int(idx[0]), int(idx[-1]) + 1
        own = s.edges[(s.edges[:, 0] >= start) & (s.edges[:, 0] < stop)] - start
        bodies.append({"kind": kind, "start": start, "stop": stop, "edges": own.tolist()})
    ny, nx = sim.mask.shape
    with open(path, "w", encoding="utf-8") as out:
        header = {"schema": STATE_SCHEMA, "nx": nx, "ny": ny, "dx_m": dx_m, "dt_s": dt_s,
                  "bodies": bodies}
        out.write(json.dumps(header) + "\n")
        for frame in range(frames + 1):
            if frame:
                sim = sim.advance(steps_per_frame)
            out.write(json.dumps({"step": sim.step, "points": sim.X.round(5).tolist()}) + "\n")
    return sim
