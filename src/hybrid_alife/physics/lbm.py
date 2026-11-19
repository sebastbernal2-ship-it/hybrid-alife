"""Lattice-Boltzmann (D2Q9) flow solver; replaces proxy flow when requested."""

from __future__ import annotations

import ctypes
import subprocess
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
