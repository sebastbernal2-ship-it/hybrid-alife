"""Metric computations for survival, diversity, communication, enrichment."""

from __future__ import annotations

import numpy as np

from hybrid_alife.metrics.bedau import hill_number_1d, lineage_hill1d
from hybrid_alife.types import SimState


def survival_fraction_embodied(state: SimState) -> float:
    if state.embodied is None:
        return 0.0
    return float(state.embodied.alive.mean())


def survival_fraction_avida(state: SimState) -> float:
    if state.avida is None:
        return 0.0
    return float(state.avida.alive.mean())


def mean_lineage_depth_embodied(state: SimState) -> float:
    if state.embodied is None:
        return 0.0
    return float(state.embodied.lineage_depth.mean())


def mean_lineage_depth_avida(state: SimState) -> float:
    if state.avida is None:
        return 0.0
    return float(state.avida.lineage_depth.mean())


def max_lineage_depth_embodied(state: SimState) -> int:
    if state.embodied is None:
        return 0
    return int(state.embodied.lineage_depth.max())


def reproductive_success_embodied(state: SimState) -> float:
    """Fraction of starting lineages still alive (proxy)."""
    if state.embodied is None:
        return 0.0
    living_lineages = np.asarray(state.embodied.lineage_id)[
        np.asarray(state.embodied.alive)
    ]
    return float(len(np.unique(living_lineages)))


def action_entropy(state: SimState) -> float:
    """Entropy over action-history indices across the population (behavioral diversity proxy)."""
    if state.embodied is None:
        return 0.0
    hist = np.asarray(state.embodied.action_history).reshape(-1)
    if hist.size == 0:
        return 0.0
    counts = np.bincount(hist, minlength=10).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def message_energy(state: SimState) -> float:
    if state.embodied is None:
        return 0.0
    return float(np.mean(np.square(np.asarray(state.embodied.messages))))


def communication_usage_rate(state: SimState) -> float:
    """Fraction of agents emitting nontrivial message energy this step."""
    if state.embodied is None:
        return 0.0
    msg = np.asarray(state.embodied.messages)
    norms = np.linalg.norm(msg, axis=-1)
    return float(np.mean(norms > 0.05))


def behavior_descriptor_array(state: SimState) -> np.ndarray:
    if state.embodied is None:
        return np.zeros((0, 2), dtype=np.float32)
    alive = np.asarray(state.embodied.alive)
    bd = np.asarray(state.embodied.behavior_descriptor)
    return bd[alive].astype(np.float32)


def enrichment_separation_index(state: SimState) -> float:
    """Proxy separation metric: high when enrichment field is spatially structured.

    Inspired by microfluidic separation -- a more bimodal/structured enrichment
    field is "more separated".
    """
    field = np.asarray(state.world.enrichment[..., 0])
    mu = np.mean(np.abs(field))
    return float(np.std(field) / (mu + 1e-6))


def coordinated_behavior_index(state: SimState) -> float:
    """Proxy convention/ritual metric: do living agents share action distribution?

    Computes the negative entropy of the most-recent action index across the
    population (high = coordinated).
    """
    if state.embodied is None:
        return 0.0
    alive = np.asarray(state.embodied.alive)
    if alive.sum() == 0:
        return 0.0
    most_recent = np.asarray(state.embodied.action_history)[alive, -1]
    counts = np.bincount(most_recent, minlength=10).astype(np.float64)
    p = counts / max(counts.sum(), 1.0)
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(np.log(max(len(counts), 1)) - ent)


def embodied_lineage_hill1d(state: SimState) -> float:
    """Effective number of lineages (Hill 1D) for living embodied agents."""
    if state.embodied is None:
        return 0.0
    return lineage_hill1d(state.embodied.lineage_id, state.embodied.alive)


def avida_lineage_hill1d(state: SimState) -> float:
    if state.avida is None:
        return 0.0
    return lineage_hill1d(state.avida.lineage_id, state.avida.alive)


def avida_tasks_solved(state: SimState) -> float:
    if state.avida is None or state.avida.tasks_completed is None:
        return 0.0
    alive = np.asarray(state.avida.alive).astype(bool)
    tc = np.asarray(state.avida.tasks_completed)[alive]
    if tc.size == 0:
        return 0.0
    return float(np.mean([bin(int(t)).count("1") for t in tc]))


def mean_avida_merit(state: SimState) -> float:
    if state.avida is None:
        return 0.0
    alive = np.asarray(state.avida.alive)
    if alive.sum() == 0:
        return 0.0
    return float(np.asarray(state.avida.merit)[alive].mean())


def collect_full_metrics(state: SimState) -> dict[str, float]:
    return {
        "generation": int(state.generation),
        "step": int(state.step),
        "embodied_alive_frac": survival_fraction_embodied(state),
        "avida_alive_frac": survival_fraction_avida(state),
        "embodied_mean_lineage_depth": mean_lineage_depth_embodied(state),
        "embodied_max_lineage_depth": max_lineage_depth_embodied(state),
        "avida_mean_lineage_depth": mean_lineage_depth_avida(state),
        "embodied_reproductive_lineages": reproductive_success_embodied(state),
        "action_entropy": action_entropy(state),
        "message_energy": message_energy(state),
        "comm_usage_rate": communication_usage_rate(state),
        "enrichment_separation": enrichment_separation_index(state),
        "coordinated_behavior_index": coordinated_behavior_index(state),
        "mean_enrichment": float(np.asarray(state.world.enrichment).mean()),
        "mean_concentration": float(np.asarray(state.world.concentration).mean()),
        "mean_metabolite": float(np.asarray(state.world.metabolites).mean()),
        "mean_avida_merit": mean_avida_merit(state),
        "embodied_lineage_hill1d": embodied_lineage_hill1d(state),
        "avida_lineage_hill1d": avida_lineage_hill1d(state),
        "avida_tasks_solved": avida_tasks_solved(state),
    }


def collect_smoke_metrics(state: SimState) -> dict[str, float | int]:
    embodied_alive = int(state.embodied.alive.sum()) if state.embodied is not None else 0
    avida_alive = int(state.avida.alive.sum()) if state.avida is not None else 0
    return {
        "embodied_alive": embodied_alive,
        "avida_alive": avida_alive,
        "mean_enrichment": float(state.world.enrichment.mean()),
        "mean_concentration": float(state.world.concentration.mean()),
    }
