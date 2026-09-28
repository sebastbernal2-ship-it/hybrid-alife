"""Runtime backend selection with an explicit GPU-to-CPU fallback."""

from __future__ import annotations

import warnings

import jax


def resolve_backend(requested: str = "auto") -> str:
    """Return a usable backend name without failing when GPU support is absent."""
    requested = requested.lower()
    if requested not in {"auto", "cpu", "gpu", "cuda"}:
        raise ValueError("backend must be one of: auto, cpu, gpu, cuda")
    if requested == "cpu":
        return "cpu"
    if requested == "auto":
        return jax.default_backend()
    try:
        devices = jax.devices("gpu")
    except RuntimeError as exc:
        warnings.warn(
            f"GPU backend unavailable ({exc}); falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    if not devices:
        warnings.warn(
            "GPU backend has no devices; falling back to CPU.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cpu"
    return "gpu"


def backend_report(requested: str = "auto") -> dict[str, str]:
    """Return requested, resolved, and active backend names for run metadata."""
    resolved = resolve_backend(requested)
    return {
        "requested_backend": requested,
        "resolved_backend": resolved,
        "active_backend": jax.default_backend(),
    }
