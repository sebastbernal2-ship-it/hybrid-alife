"""Lattice-Boltzmann (D2Q9) flow solver; replaces proxy flow when requested.

Frame convention
----------------
Fields are [rows, cols, ...] arrays. Two frames exist, and only two:

- **Solver frame** (``solve``, ``channel_mask``, ``channel_flow``, ``serpentine_mask``,
  ``LiveLBM.f/mask/u/v/rho``): component 0 (``u``, LBM x) is along *columns*, component 1
  (``v``, LBM y) along rows; a channel runs along columns with walls on rows.
- **World frame** (``world_channel_flow``, ``LiveLBM.flow/velocity``, agent positions, forces
  and ``pending_force``): component ``k`` is along array axis ``k``, so component 0 is along
  *rows*, which is how ``flow_movement_scale`` moves agents; a channel runs along rows with
  walls on the first and last column.

World = solver field with the two *array axes* swapped (``to_world_frame``); the components
are not swapped, because swapping the axes already turns "along columns" into "along rows".
A world [H, W] channel is therefore a solver [W, H] channel.

``flow_solver: lbm_live`` is always in the world frame. The static ``flow_solver: lbm`` keeps
the solver frame by default (``lbm_frame: legacy``, the same convention as the proxy fields,
so recorded outputs do not change) and uses the world frame with ``lbm_frame: world``.
"""

from __future__ import annotations

import ctypes
import subprocess
from dataclasses import dataclass, field, replace
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


def serpentine_mask(h: int, w: int, waves: int = 1, amplitude: float = 0.2) -> np.ndarray:
    """Sinusoidal channel in the solver frame [h, w]: solid outside a meandering lumen.

    The centreline is ``(h - 1) / 2 + amplitude * h * sin(2 pi waves x / w)``; the lumen is
    every cell within half-width ``h / 2 - 1 - amplitude * h`` of it. ``waves`` is a whole
    number so the channel closes on itself across the periodic boundary. The first and last
    rows are always solid.
    """
    if not np.isfinite(waves) or waves < 1 or int(waves) != waves:
        raise ValueError("serpentine waves must be a positive whole number (periodic channel)")
    half = h / 2 - 1 - amplitude * h
    if not np.isfinite(amplitude) or amplitude < 0.0 or half < 1.5:
        raise ValueError(
            f"serpentine amplitude {amplitude} leaves half-width {half:.2f} < 1.5 cells in "
            f"{h} rows; lower the amplitude or widen the channel"
        )
    centre = (h - 1) / 2 + amplitude * h * np.sin(2.0 * np.pi * waves * np.arange(w) / w)
    mask = (np.abs(np.arange(h)[:, None] - centre[None, :]) > half).astype(np.uint8)
    mask[0, :] = mask[-1, :] = 1
    return mask


def to_world_frame(field: np.ndarray) -> np.ndarray:
    """Solver-frame [rows, cols, ...] field as a world-frame field (see module docstring)."""
    return np.ascontiguousarray(np.swapaxes(field, 0, 1))


def world_channel_flow(h: int, w: int, **kwargs) -> np.ndarray:
    """``channel_flow`` for a world [h, w] grid, in the world frame."""
    return to_world_frame(channel_flow(w, h, **kwargs))


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
    # Law carrier (periodic box): target mean velocity (lattice units, world frame) held by a
    # uniform acceleration (target - mean) / relax_steps, a stand-in for the wall friction or
    # pump that sets a channel's flow. None keeps the legacy free-running fluid.
    target: np.ndarray = field(default=None)
    relax_steps: float = 0.0

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
        mask: np.ndarray | None = None,
    ) -> LiveLBM:
        """Channel along world axis 0, warmed up to (near) steady flow.

        ``mask`` (solver frame [width, height]) replaces the straight or obstacle channel.
        The body force is always the straight channel's for ``peak_speed``, so a narrower or
        meandering lumen peaks below ``peak_speed``.
        """
        if mask is None:
            mask = channel_mask(width, height, obstacle_radius)  # LBM frame is [W, H]
        else:
            if np.shape(mask) != (width, height):
                raise ValueError(f"mask must have shape {(width, height)}, got {np.shape(mask)}")
            mask = np.ascontiguousarray(np.asarray(mask) != 0, dtype=np.uint8)
        u, v, rho = np.zeros(mask.shape), np.zeros(mask.shape), np.ones(mask.shape)
        f = np.zeros((*mask.shape, 9))
        _load().lbm_init(mask.shape[0], mask.shape[1], 1.0, u, v, f)
        out = cls(f, mask, nu, channel_body_force(mask.shape[0], nu, peak_speed), peak_speed,
                  u, v, rho, np.zeros((height, width, 2)))
        return out.advance(warm_steps)

    @classmethod
    def uniform(
        cls,
        height: int,
        width: int,
        nu: float = 0.1,
        peak_speed: float = 0.05,
        velocity: tuple[float, float] = (0.0, 0.0),
        relax_steps: float = 0.0,
    ) -> LiveLBM:
        """Fully periodic wall-free box at uniform `velocity` (lattice units, world frame).

        A uniform flow is an exact steady state of the periodic Navier-Stokes equations, so no
        relaxation is needed: the distributions start at their equilibrium and stay there
        until something (agents) perturbs them.
        """
        mask = np.zeros((width, height), np.uint8)  # LBM frame is [W, H]
        u = np.full(mask.shape, float(velocity[0]))
        v = np.full(mask.shape, float(velocity[1]))
        f = np.zeros((*mask.shape, 9))
        _load().lbm_init(mask.shape[0], mask.shape[1], 1.0, u, v, f)
        return cls(f, mask, nu, 0.0, peak_speed, u, v, np.ones(mask.shape),
                   np.zeros((height, width, 2)), np.asarray(velocity, np.float64), relax_steps)

    def with_uniform_flow(self, velocity: tuple[float, float]) -> LiveLBM:
        """Fresh uniform fluid at `velocity` with this fluid's box, viscosity and relaxation."""
        h, w = self.shape
        return LiveLBM.uniform(h, w, self.nu, self.peak_speed, velocity, self.relax_steps)

    def mean_velocity(self) -> np.ndarray:
        fluid = self.mask == 0
        return np.array([self.u[fluid].mean(), self.v[fluid].mean()])

    def advance(self, steps: int, force: np.ndarray | None = None) -> LiveLBM:
        """Return a copy advanced by `steps` LBM steps under the body force plus `force`.

        `force` is a world-frame [H, W, 2] force density per cell, applied every step.
        Defaults to `pending_force`.
        """
        h, w = self.shape
        if force is None:
            force = self.pending_force if self.pending_force is not None else np.zeros((h, w, 2))
        force = np.asarray(force, np.float64)
        if force.shape != (h, w, 2):  # the kernel reads h*w*2 doubles: a bad shape is an overread
            raise ValueError(f"force must have shape {(h, w, 2)}, got {force.shape}")
        force_l = np.ascontiguousarray(force.transpose(1, 0, 2))
        f = np.ascontiguousarray(self.f.copy())
        u, v, rho = (np.zeros(self.mask.shape) for _ in range(3))
        bx, by = self.body_force, 0.0
        if self.target is not None and self.relax_steps > 0.0:
            gx, gy = (self.target - self.mean_velocity()) / self.relax_steps
            bx, by = bx + gx, by + gy
        rc = _load().lbm_advance(
            self.mask.shape[0], self.mask.shape[1], self.mask, self.nu, bx, by,
            force_l.ctypes.data, steps, f, u, v, rho,
        )
        if rc < 0:
            raise ValueError("invalid LBM arguments (need h,w>=3, nu>0, steps>=0)")
        speed = float(np.max(np.hypot(u, v)))
        if not np.isfinite(speed) or speed > MAX_LATTICE_SPEED:
            raise ValueError(
                f"live LBM velocity {speed:.3g} exceeds the low-Mach limit {MAX_LATTICE_SPEED} "
                "(lattice units); lower lbm_agent_drag or lbm_peak_speed, or raise lbm_viscosity"
            )
        return replace(self, f=f, u=u, v=v, rho=rho)

    def with_pending_force(self, force: np.ndarray) -> LiveLBM:
        force = np.asarray(force, np.float64)
        if force.shape != (*self.shape, 2):
            raise ValueError(f"force must have shape {(*self.shape, 2)}, got {force.shape}")
        return replace(self, pending_force=force)

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


SOLID_FORCE_POLICIES = ("drop", "redistribute")


def redistribute_solid_forces(force: np.ndarray, solid: np.ndarray) -> np.ndarray:
    """Move force sitting on solid cells to nearby fluid cells; the total is unchanged.

    ``force`` is [H, W, 2] and ``solid`` [H, W], same frame. Each solid cell's force is split
    equally among the fluid cells on the nearest square ring (Chebyshev distance 1, then 2,
    ...) that holds any, wrapping periodically as the LBM grid does. Fluid-cell forces stay.
    """
    solid = np.asarray(solid) != 0
    if solid.all():
        raise ValueError("no fluid cell to receive the force")
    h, w = solid.shape
    out = np.where(solid[..., None], 0.0, force)
    for y, x in zip(*np.nonzero(solid & np.any(force != 0, axis=-1)), strict=True):
        for r in range(1, max(h, w) + 1):
            span = range(-r, r + 1)
            # A set: on small grids several offsets wrap to the same cell.
            ring = sorted({
                ((y + dy) % h, (x + dx) % w)
                for dy in span for dx in span if max(abs(dy), abs(dx)) == r
            })
            targets = [c for c in ring if not solid[c]]
            if targets:
                for c in targets:
                    out[c] += force[y, x] / len(targets)
                break
    return out


def _check_policy(solid_policy: str) -> None:
    if solid_policy not in SOLID_FORCE_POLICIES:
        raise ValueError(f"unknown solid_policy={solid_policy!r} (drop or redistribute)")


def agent_reaction_forces(
    positions: np.ndarray,
    swim_velocities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    solid_world: np.ndarray,
    coeff: float,
    solid_policy: str = "drop",
) -> np.ndarray:
    """Per-cell force density the agents exert on the fluid (world frame, [H, W, 2]).

    `swim_velocities` are agent velocities relative to the local fluid, in lattice units
    (cells per LBM step), so passive advection carries no slip. Each agent feels Stokes
    drag -coeff * radius * v_swim; by Newton's third law it pushes the fluid with
    +coeff * radius * v_swim at its cell. Slip is clipped to MAX_RELATIVE_SPEED for
    stability. Dead agents deposit nothing. Agents on solid cells (movement does not collide
    with walls) deposit nothing under ``solid_policy="drop"``; under ``"redistribute"`` their
    force goes to the nearest fluid cells (see ``redistribute_solid_forces``).
    """
    _check_policy(solid_policy)
    h, w = solid_world.shape
    ry = np.clip(np.floor(positions[:, 0] * h).astype(int), 0, h - 1)
    cx = np.clip(np.floor(positions[:, 1] * w).astype(int), 0, w - 1)
    slip = np.clip(swim_velocities, -MAX_RELATIVE_SPEED, MAX_RELATIVE_SPEED)
    force = coeff * radii[:, None] * slip
    keep = np.asarray(alive, bool)
    if solid_policy == "drop":
        keep = keep & (solid_world[ry, cx] == 0)
    force = force * keep[:, None]
    out = np.zeros((h, w, 2))
    np.add.at(out, (ry, cx), force)
    return out if solid_policy == "drop" else redistribute_solid_forces(out, solid_world)


def agent_dipole_forces(
    positions: np.ndarray,
    swim_velocities: np.ndarray,
    radii: np.ndarray,
    alive: np.ndarray,
    solid_world: np.ndarray,
    coeff: float,
    length: float,
    toroidal: bool,
    solid_policy: str = "drop",
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
    With ``solid_policy="redistribute"`` a solid end no longer skips the agent: both ends are
    deposited and any force on a solid cell moves to the nearest fluid cells, which keeps the
    net force zero (the pair's separation is then no longer guaranteed even).
    """
    _check_policy(solid_policy)
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
    ok = np.asarray(alive, bool) & (speed > 0) & inside & np.any(offset != 0, axis=1)
    if solid_policy == "drop":
        ok &= (solid_world[front[:, 0], front[:, 1]] == 0) & (
            solid_world[rear[:, 0], rear[:, 1]] == 0
        )
    push = (magnitude[:, None] * unit) * ok[:, None]
    np.add.at(out, (front[:, 0], front[:, 1]), push)
    np.add.at(out, (rear[:, 0], rear[:, 1]), -push)
    return out if solid_policy == "drop" else redistribute_solid_forces(out, solid_world)
