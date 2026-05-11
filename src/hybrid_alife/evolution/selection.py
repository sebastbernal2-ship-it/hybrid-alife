"""Selection skeletons for v1 implementation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tournament_select(fitness: jax.Array, tournament_size: int, key: jax.Array) -> jax.Array:
    """Return one parent index per individual using vectorized tournament selection."""
    n = fitness.shape[0]
    candidates = jax.random.randint(key, (n, tournament_size), 0, n)
    candidate_fitness = fitness[candidates]
    winners = jnp.argmax(candidate_fitness, axis=1)
    return jnp.take_along_axis(candidates, winners[:, None], axis=1)[:, 0]

