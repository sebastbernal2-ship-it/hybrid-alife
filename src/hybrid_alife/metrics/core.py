"""Metric skeletons for survival, diversity, communication, and enrichment."""

from __future__ import annotations

import jax.numpy as jnp

from hybrid_alife.types import SimState


def enrichment_separation_index(state: SimState) -> float:
    """Proxy separation metric: high when enrichment field is spatially structured."""
    field = state.world.enrichment[..., 0]
    return float(jnp.std(field) / (jnp.mean(jnp.abs(field)) + 1e-6))


def embodied_message_energy(state: SimState) -> float:
    if state.embodied is None:
        return 0.0
    return float(jnp.mean(jnp.square(state.embodied.messages)))

