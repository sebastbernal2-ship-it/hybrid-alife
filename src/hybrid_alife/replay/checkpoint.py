"""Deterministic replay checkpointing.

Checkpoints are pickled dicts that contain enough state to resume:
  - config dict (raw yaml)
  - generation, step, rng
  - world arrays (numpy)
  - embodied + avida arrays (numpy)
  - metrics summary
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np

from hybrid_alife.types import (
    AvidaPopulationState,
    EmbodiedPopulationState,
    SimState,
    WorldState,
)


def _to_numpy(x: Any) -> Any:
    if hasattr(x, "shape") and hasattr(x, "dtype") and not isinstance(x, np.ndarray):
        return np.asarray(x)
    return x


def _state_dict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    out = {}
    for k, v in asdict(obj).items() if is_dataclass(obj) else vars(obj).items():
        if isinstance(v, dict):
            out[k] = {kk: _to_numpy(vv) for kk, vv in v.items()}
        else:
            out[k] = _to_numpy(v)
    return out


def save_checkpoint(path: str | Path, state: SimState, config: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config,
        "generation": int(state.generation),
        "step": int(state.step),
        "rng": _to_numpy(state.rng),
        "world": _state_dict(state.world),
        "embodied": _state_dict(state.embodied) if state.embodied is not None else None,
        "avida": _state_dict(state.avida) if state.avida is not None else None,
        "metrics": dict(state.metrics),
    }
    with path.open("wb") as f:
        pickle.dump(payload, f)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as f:
        return pickle.load(f)


def restore_states(
    payload: dict[str, Any],
) -> tuple[WorldState, EmbodiedPopulationState | None, AvidaPopulationState | None]:
    """Rebuild dataclass instances from a checkpoint payload."""
    import jax.numpy as jnp

    def to_jax(v: Any) -> Any:
        if isinstance(v, np.ndarray):
            return jnp.asarray(v)
        if isinstance(v, dict):
            return {kk: to_jax(vv) for kk, vv in v.items()}
        return v

    world_kwargs = {k: to_jax(v) for k, v in payload["world"].items()}
    world = WorldState(**world_kwargs)

    emb_payload = payload.get("embodied")
    if emb_payload:
        emb_kwargs = {k: to_jax(v) for k, v in emb_payload.items()}
        embodied = EmbodiedPopulationState(**emb_kwargs)
    else:
        embodied = None

    av_payload = payload.get("avida")
    if av_payload:
        av_kwargs = {k: to_jax(v) for k, v in av_payload.items()}
        avida = AvidaPopulationState(**av_kwargs)
    else:
        avida = None

    return world, embodied, avida
