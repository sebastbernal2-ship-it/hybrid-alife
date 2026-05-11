"""Quality-diversity / MAP-Elites summary metrics.

Augments the existing MapElitesArchive with the headline numbers expected by
the QD literature: QD-score, coverage, and Shannon archive entropy over
occupied cells (a mode-collapse detector recommended by the validation memo).
"""

from __future__ import annotations

import numpy as np

from hybrid_alife.evolution.archives import MapElitesArchive


def qd_score(archive: MapElitesArchive) -> float:
    """Sum of fitness over filled cells.

    Following Mouret & Clune (2015, arXiv:1504.04909), QD-score sums positive
    contributions only; we shift fitness so all filled cells contribute
    >= 0 to avoid penalising rare cells with negative fitness.
    """
    if not archive.grid_filled.any():
        return 0.0
    fills = archive.grid_fitness[archive.grid_filled]
    return float(np.maximum(fills - fills.min(), 0.0).sum())


def coverage(archive: MapElitesArchive) -> float:
    return float(archive.grid_filled.mean())


def archive_entropy(archive: MapElitesArchive) -> float:
    """Shannon entropy over the normalized fitness distribution of occupied cells.

    A flat (uniformly-good) archive has high entropy; a single-peak archive
    has low entropy. Reported in nats.
    """
    if not archive.grid_filled.any():
        return 0.0
    fills = archive.grid_fitness[archive.grid_filled].astype(np.float64)
    shifted = fills - fills.min() + 1e-9
    p = shifted / shifted.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def occupancy_entropy(archive: MapElitesArchive) -> float:
    """Shannon entropy over cell-presence (purely diversity, not quality)."""
    filled = archive.grid_filled.astype(np.float64)
    s = filled.sum()
    if s <= 0:
        return 0.0
    p = filled.flatten() / s
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def qd_summary(archive: MapElitesArchive) -> dict[str, float]:
    return {
        "qd_score": qd_score(archive),
        "coverage": coverage(archive),
        "archive_entropy": archive_entropy(archive),
        "occupancy_entropy": occupancy_entropy(archive),
    }
