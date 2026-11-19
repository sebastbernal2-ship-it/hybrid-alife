"""Lattice-Boltzmann (D2Q9) flow solver; replaces proxy flow when requested."""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_LIB_PATH = Path(__file__).with_name("liblbm.so")
_NATIVE_DIR = Path(__file__).resolve().parents[3] / "native"
_lib: ctypes.CDLL | None = None
# D2Q9 is only valid for lattice speeds well below c_s = 1/sqrt(3); beyond this the result is
# compressibility noise, not flow, even when it is finite.
MAX_LATTICE_SPEED = 0.3


def _load() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        if not _LIB_PATH.exists():
            try:
                subprocess.run(
                    ["make", "-C", str(_NATIVE_DIR)], check=True, capture_output=True, text=True
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                detail = getattr(exc, "stderr", None) or str(exc)
                raise RuntimeError(
                    "building the native LBM kernel failed (needs g++ and make; "
                    f"run `make -C native`):\n{detail}"
                ) from exc
        lib = ctypes.CDLL(str(_LIB_PATH))
        dp = np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS")
        lib.lbm_solve.argtypes = [
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_int, dp, dp, dp,
        ]
        lib.lbm_solve.restype = ctypes.c_int
        lib.lbm_init.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_double, dp, dp, dp]
        lib.lbm_init.restype = None
        lib.lbm_advance.argtypes = [
            ctypes.c_int, ctypes.c_int,
            np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS"),
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_void_p,
            ctypes.c_int, dp, dp, dp, dp,
        ]
        lib.lbm_advance.restype = ctypes.c_int
        _lib = lib
    return _lib


def solve(
    mask: np.ndarray, nu: float, force: tuple[float, float], steps: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run LBM to (near) steady state. `mask` is [H, W], nonzero = solid.

    Returns (u, v, rho), each [H, W] float64, in lattice units.
    """
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D [H, W], got shape {mask.shape}")
    mask = np.ascontiguousarray(mask != 0, dtype=np.uint8)
    h, w = mask.shape
    u, v, rho = (np.zeros((h, w), np.float64) for _ in range(3))
    rc = _load().lbm_solve(h, w, mask, nu, force[0], force[1], steps, u, v, rho)
    if rc < 0:
        raise ValueError("invalid LBM arguments (need h,w>=3, nu>0, steps>=0)")
    speed = float(np.max(np.hypot(u, v))) if u.size else 0.0
    if not np.isfinite(speed) or speed > MAX_LATTICE_SPEED:
        raise ValueError(
            f"LBM velocity {speed:.3g} exceeds the low-Mach limit {MAX_LATTICE_SPEED} "
            "(lattice units); reduce the force or raise the viscosity"
        )
    return u, v, rho


def channel_mask(h: int, w: int, obstacle_radius: float = 0.0) -> np.ndarray:
    """Walls on the top/bottom rows, optional circular obstacle at the centre."""
    mask = np.zeros((h, w), np.uint8)
    mask[0, :] = mask[-1, :] = 1
    if obstacle_radius > 0:
        yy, xx = np.mgrid[0:h, 0:w]
        mask[(yy - h / 2) ** 2 + (xx - w / 4) ** 2 <= obstacle_radius**2] = 1
    return mask


def channel_flow(
    h: int, w: int, nu: float = 0.1, force: float = 1e-5, steps: int = 4000,
    obstacle_radius: float = 0.0,
) -> np.ndarray:
    """Steady channel flow as a [H, W, 2] float32 (u, v) field normalised to unit peak speed."""
    u, v, _ = solve(channel_mask(h, w, obstacle_radius), nu, (force, 0.0), steps)
    uv = np.stack([u, v], -1)
    peak = np.abs(uv).max()
    return (uv / peak if peak > 0 else uv).astype(np.float32)


_CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
_CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
MAX_RELATIVE_SPEED = 0.1  # lattice units; larger agent-fluid slip risks LBM instability


def channel_body_force(height: int, nu: float, peak_speed: float) -> float:
    """Body acceleration whose steady Poiseuille flow peaks at `peak_speed` (lattice units)."""
    return 8.0 * nu * peak_speed / (height - 2) ** 2


@dataclass
class LiveLBM:
    """Stateful LBM fluid in the *world* frame.

    World arrays are [H, W, ...] with vector component 0 along axis 0 (rows), matching how
    `flow_movement_scale` moves agents. Internally the solver runs on the transposed grid
    (LBM x = world axis 0), so the channel flows along axis 0 with walls on the first and
    last column. State is plain numpy (`f`, `mask`, ...), so it pickles with checkpoints and
    carries no native handle. Methods return new objects; the input is never mutated.
    """

    f: np.ndarray  # [W, H, 9] distributions in the LBM frame
    mask: np.ndarray  # [W, H] uint8 solid mask in the LBM frame
    nu: float
    body_force: float
    peak_speed: float
    u: np.ndarray  # [W, H] last macroscopic fields, LBM frame
    v: np.ndarray
    rho: np.ndarray
    pending_force: np.ndarray = field(default=None)  # [H, W, 2] world frame, per cell

    @property
    def shape(self) -> tuple[int, int]:
        return self.f.shape[1], self.f.shape[0]  # world (H, W)

    @classmethod
    def channel(
        cls,
        height: int,
        width: int,
        nu: float = 0.1,
        peak_speed: float = 0.05,
        obstacle_radius: float = 0.0,
        warm_steps: int = 3000,
    ) -> LiveLBM:
        """Channel along world axis 0, warmed up to (near) steady flow."""
        mask = channel_mask(width, height, obstacle_radius)  # LBM frame is [W, H]
        n = mask.size
        u, v, rho = np.zeros(mask.shape), np.zeros(mask.shape), np.ones(mask.shape)
        f = np.zeros((*mask.shape, 9))
        _load().lbm_init(mask.shape[0], mask.shape[1], 1.0, u, v, f)
        out = cls(f, mask, nu, channel_body_force(mask.shape[0], nu, peak_speed), peak_speed,
                  u, v, rho, np.zeros((height, width, 2)))
        assert n == height * width
        return out.advance(warm_steps)

    def advance(self, steps: int, force: np.ndarray | None = None) -> LiveLBM:
        """Return a copy advanced by `steps` LBM steps under the body force plus `force`.

        `force` is a world-frame [H, W, 2] force density per cell, applied every step.
        Defaults to `pending_force`.
        """
        force = self.pending_force if force is None else force
        force_l = np.ascontiguousarray(np.asarray(force, np.float64).transpose(1, 0, 2))
        f = np.ascontiguousarray(self.f.copy())
        u, v, rho = (np.zeros(self.mask.shape) for _ in range(3))
        rc = _load().lbm_advance(
            self.mask.shape[0], self.mask.shape[1], self.mask, self.nu, self.body_force, 0.0,
            force_l.ctypes.data, steps, f, u, v, rho,
        )
        if rc < 0:
            raise ValueError("invalid LBM arguments (need h,w>=3, nu>0, steps>=0)")
        return LiveLBM(f, self.mask, self.nu, self.body_force, self.peak_speed, u, v, rho,
                       self.pending_force)

    def with_pending_force(self, force: np.ndarray) -> LiveLBM:
        return LiveLBM(self.f, self.mask, self.nu, self.body_force, self.peak_speed, self.u,
                       self.v, self.rho, np.asarray(force, np.float64))

    def velocity(self) -> np.ndarray:
        """[H, W, 2] fluid velocity in lattice units (world frame)."""
        return np.stack([self.u.T, self.v.T], axis=-1)

    def flow(self) -> np.ndarray:
        """[H, W, 2] float32 flow in world-flow units (steady channel peak is about 1)."""
        return (self.velocity() / self.peak_speed).astype(np.float32)

    def momentum(self) -> np.ndarray:
        """Raw fluid momentum sum(f * c) over fluid cells, as (axis 0, axis 1)."""
        fluid = (self.mask == 0)[..., None]
        px = np.sum(self.f * _CX * fluid)
        py = np.sum(self.f * _CY * fluid)
        return np.array([px, py])


def agent_reaction_forces(
    positions: np.ndarray,
    swim_velocities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    solid_world: np.ndarray,
    coeff: float,
) -> np.ndarray:
    """Per-cell force density the agents exert on the fluid (world frame, [H, W, 2]).

    `swim_velocities` are agent velocities relative to the local fluid, in lattice units
    (cells per LBM step), so passive advection carries no slip. Each agent feels Stokes
    drag -coeff * radius * v_swim; by Newton's third law it pushes the fluid with
    +coeff * radius * v_swim at its cell. Slip is clipped to MAX_RELATIVE_SPEED for
    stability. Dead agents and agents on solid cells deposit nothing.
    """
    h, w = solid_world.shape
    ry = np.clip(np.floor(positions[:, 0] * h).astype(int), 0, h - 1)
    cx = np.clip(np.floor(positions[:, 1] * w).astype(int), 0, w - 1)
    slip = np.clip(swim_velocities, -MAX_RELATIVE_SPEED, MAX_RELATIVE_SPEED)
    force = coeff * radii[:, None] * slip
    force = force * (np.asarray(alive, bool) & (solid_world[ry, cx] == 0))[:, None]
    out = np.zeros((h, w, 2))
    np.add.at(out, (ry, cx), force)
    return out


def agent_dipole_forces(
    positions: np.ndarray,
    swim_velocities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    solid_world: np.ndarray,
    coeff: float,
    length: float,
    toroidal: bool,
) -> np.ndarray:
    """Force-free (pusher) dipole version of `agent_reaction_forces`, [H, W, 2].

    Each swimming agent deposits +F e at the cell n ahead of its own cell and -F e at the
    cell n behind it, with e the unit swim direction, n = round(length * e / 2) per axis,
    and F = coeff * radius * |clipped swim velocity|. The net force per agent is exactly
    zero, and the cell separation 2n is always even along every axis. That is deliberate:
    a pair separated by an odd number of nodes projects onto the lattice's grid-scale
    (Nyquist) checkerboard mode, which D2Q9 BGK does not damp, and leaves persistent
    spurious flow across the whole domain (measured: about 300x more spectral energy at
    k near Nyquist than an even gap). So the effective dipole length is quantized to
    2 * round(length * e / 2) cells.

    Agents are skipped entirely (never half-deposited, which would break the zero net
    force) when either point lands on a solid cell or outside a non-toroidal domain, when
    n rounds to zero (both points on the agent's cell cancel), or when not swimming.
    """
    h, w = solid_world.shape
    out = np.zeros((h, w, 2))
    slip = np.clip(swim_velocities, -MAX_RELATIVE_SPEED, MAX_RELATIVE_SPEED)
    speed = np.linalg.norm(slip, axis=1)
    unit = slip / np.maximum(speed, 1e-12)[:, None]
    magnitude = coeff * radii * speed
    cell = np.floor(np.asarray(positions) * np.array([h, w])).astype(int)
    cell = np.clip(cell, 0, [h - 1, w - 1])
    offset = np.rint(0.5 * length * unit).astype(int)
    ends, inside = [], np.ones(cell.shape[0], bool)
    for sign in (1, -1):
        point = cell + sign * offset
        if toroidal:
            point = np.mod(point, [h, w])
        inside &= np.all((point >= 0) & (point < [h, w]), axis=1)
        ends.append(np.clip(point, 0, [h - 1, w - 1]))
    front, rear = ends
    ok = (
        np.asarray(alive, bool) & (speed > 0) & inside
        & (solid_world[front[:, 0], front[:, 1]] == 0)
        & (solid_world[rear[:, 0], rear[:, 1]] == 0)
        & np.any(offset != 0, axis=1)
    )
    push = (magnitude[:, None] * unit) * ok[:, None]
    np.add.at(out, (front[:, 0], front[:, 1]), push)
    np.add.at(out, (rear[:, 0], rear[:, 1]), -push)
    return out
