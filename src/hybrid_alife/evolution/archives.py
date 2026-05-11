"""Novelty archive and MAP-Elites archive.

Both store behavior descriptors (2D) plus a scalar fitness. They are intentionally
simple in-memory numpy arrays so they can be checkpointed easily and used from
non-jitted code.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class NoveltyArchive:
    capacity: int
    dim: int = 2
    k: int = 5
    descriptors: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.float32))

    def add_batch(self, batch: np.ndarray) -> None:
        if batch.size == 0:
            return
        combined = np.concatenate([self.descriptors, batch.astype(np.float32)], axis=0)
        if combined.shape[0] > self.capacity:
            # Keep most recent
            combined = combined[-self.capacity :]
        self.descriptors = combined

    def novelty(self, batch: np.ndarray) -> np.ndarray:
        """Return per-row k-NN distance against the archive."""
        if self.descriptors.shape[0] == 0:
            return np.zeros((batch.shape[0],), dtype=np.float32)
        # pairwise distances
        diff = batch[:, None, :] - self.descriptors[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        k = min(self.k, dist.shape[1])
        top = np.partition(dist, kth=k - 1, axis=1)[:, :k]
        return top.mean(axis=1).astype(np.float32)


@dataclass
class MapElitesArchive:
    bins: int
    bounds: tuple[float, float] = (0.0, 1.0)
    grid_fitness: np.ndarray = field(init=False)
    grid_descriptor: np.ndarray = field(init=False)
    grid_filled: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.grid_fitness = np.full((self.bins, self.bins), -np.inf, dtype=np.float32)
        self.grid_descriptor = np.zeros((self.bins, self.bins, 2), dtype=np.float32)
        self.grid_filled = np.zeros((self.bins, self.bins), dtype=bool)

    def _to_bin(self, descriptors: np.ndarray) -> np.ndarray:
        lo, hi = self.bounds
        scaled = (descriptors - lo) / max(hi - lo, 1e-9)
        idx = np.clip((scaled * self.bins).astype(int), 0, self.bins - 1)
        return idx

    def update(self, descriptors: np.ndarray, fitness: np.ndarray) -> None:
        if descriptors.size == 0:
            return
        idx = self._to_bin(descriptors)
        for i in range(descriptors.shape[0]):
            r, c = int(idx[i, 0]), int(idx[i, 1])
            f = float(fitness[i])
            if f > self.grid_fitness[r, c]:
                self.grid_fitness[r, c] = f
                self.grid_descriptor[r, c] = descriptors[i]
                self.grid_filled[r, c] = True

    @property
    def coverage(self) -> float:
        return float(self.grid_filled.mean())

    @property
    def qd_score(self) -> float:
        return float(np.where(self.grid_filled, self.grid_fitness, 0.0).sum())
