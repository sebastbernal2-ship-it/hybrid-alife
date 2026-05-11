"""Bedau-Packard evolutionary activity statistics.

Implements the three core activity statistics from Bedau & Packard and the
paired-shadow (neutral) framework reviewed in Bullock & Bedau (2006,
*Artificial Life*). The "activity" of a component is the count of generations
on which it persists with non-zero abundance / usage; activity is "adaptive"
relative to a neutral-shadow run computed under the same demographics but with
selection switched off.

Components here are abstract — anything from genome hashes, behavior
descriptor cell indices, lineage ids, to logic-task ids will do. The caller
provides per-generation `present_set` and `usage_count` dicts; the bookkeeper
turns them into A_new, A_cum, A_p activity vectors.

References
----------
- Bullock & Bedau, "Exploring adaptation with evolutionary activity plots",
  *Artificial Life* 12 (2006): 193-197. https://people.reed.edu/~mab/publications/papers/bullock.bedau.ALJ06.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Hashable

import numpy as np


@dataclass
class ActivityTracker:
    """Per-component activity accumulator.

    For each component id we keep:
      - first_gen: generation in which the component was first observed
      - last_gen: most recent generation in which it was active
      - cum: cumulative usage count

    The standard Bedau activity for a component at generation g is the
    cumulative usage so far (so A_cum is sum of activities; A_new counts
    components whose first_gen == g; A_p counts components with last_gen == g
    and lifespan >= persistence_threshold).
    """

    persistence_threshold: int = 5
    first_gen: dict[Hashable, int] = field(default_factory=dict)
    last_gen: dict[Hashable, int] = field(default_factory=dict)
    cum: dict[Hashable, float] = field(default_factory=dict)

    def observe(self, generation: int, usage: dict[Hashable, float]) -> None:
        for c, count in usage.items():
            if count <= 0:
                continue
            if c not in self.first_gen:
                self.first_gen[c] = generation
            self.last_gen[c] = generation
            self.cum[c] = self.cum.get(c, 0.0) + float(count)

    def stats(self, generation: int) -> dict[str, float]:
        """Per-generation Bedau activity statistics."""
        a_new = sum(1 for c, g in self.first_gen.items() if g == generation)
        a_cum = float(sum(self.cum.values()))
        a_p = sum(
            1
            for c in self.first_gen
            if self.last_gen.get(c, -1) == generation
            and (generation - self.first_gen[c]) >= self.persistence_threshold
        )
        return {
            "A_new": float(a_new),
            "A_cum": float(a_cum),
            "A_p": float(a_p),
            "component_count": float(len(self.first_gen)),
        }


def adaptive_activity(
    experimental: list[dict[str, float]], shadow: list[dict[str, float]]
) -> list[dict[str, float]]:
    """Pair-wise subtraction: experimental - shadow per generation."""
    out: list[dict[str, float]] = []
    n = min(len(experimental), len(shadow))
    for i in range(n):
        row = {
            f"adaptive_{k}": experimental[i][k] - shadow[i].get(k, 0.0)
            for k in experimental[i]
        }
        out.append(row)
    return out


def hill_number_1d(counts: np.ndarray) -> float:
    """Hill number of order 1 (effective species count).

    H1 = exp(Shannon entropy) over a normalized count vector. Robust to
    rare-class noise — recommended in the memo for effective lineage count.
    """
    counts = np.asarray(counts, dtype=np.float64)
    counts = counts[counts > 0]
    if counts.size == 0:
        return 0.0
    p = counts / counts.sum()
    shannon = -np.sum(p * np.log(p))
    return float(np.exp(shannon))


def lineage_hill1d(lineage_ids: np.ndarray, alive: np.ndarray) -> float:
    """Effective number of lineages among the living population."""
    ids = np.asarray(lineage_ids)
    a = np.asarray(alive).astype(bool)
    if a.sum() == 0:
        return 0.0
    living = ids[a]
    _, counts = np.unique(living, return_counts=True)
    return hill_number_1d(counts)
