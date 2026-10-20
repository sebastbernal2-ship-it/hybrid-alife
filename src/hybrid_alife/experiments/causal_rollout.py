"""Frozen simulator rollouts and metrics for causal-inquiry proxies.

These metrics keep evaluator-only labels separate from controller observations.
They are proxy evidence until the simulator gains an explicit hidden-rule task.
"""

from __future__ import annotations

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.field_discovery import (
    _copy_trained_policy,
    _evaluation_flow,
)
from hybrid_alife.experiments.runner import initialize_sim, step_sim
from hybrid_alife.world.env import positions_to_grid


def _alignment(
    moves: np.ndarray,
    flows: np.ndarray,
    alive: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    move_values = np.asarray(moves, dtype=np.float64)
    flow_values = np.asarray(flows, dtype=np.float64)
    if move_values.shape != flow_values.shape or move_values.ndim != 3:
        raise ValueError("moves and flows must have the same [steps, agents, 2] shape")
    if alive is None:
        valid_alive = np.ones(move_values.shape[:2], dtype=bool)
    else:
        valid_alive = np.asarray(alive, dtype=bool)
        if valid_alive.shape != move_values.shape[:2]:
            raise ValueError("alive must have shape [steps, agents]")
    move_norm = np.linalg.norm(move_values, axis=-1)
    flow_norm = np.linalg.norm(flow_values, axis=-1)
    valid = valid_alive & (move_norm > 1e-8) & (flow_norm > 1e-8)
    values = np.sum(move_values * flow_values, axis=-1) / np.maximum(
        move_norm * flow_norm, 1e-8
    )
    return values, valid


def prediction_update_proxy(
    moves: np.ndarray,
    flows: np.ndarray,
    probe_gates: np.ndarray,
    alive: np.ndarray | None = None,
    *,
    window: int = 2,
) -> dict[str, float | int | bool | None]:
    """Measure alignment before and after each agent's first probe event."""
    if window < 1:
        raise ValueError("window must be positive")
    alignments, valid = _alignment(moves, flows, alive)
    gates = np.asarray(probe_gates, dtype=np.float64)
    if gates.shape != alignments.shape:
        raise ValueError("probe_gates must have shape [steps, agents]")
    pre_values: list[float] = []
    post_values: list[float] = []
    probe_events = 0
    for agent in range(gates.shape[1]):
        probe_indices = np.flatnonzero(gates[:, agent] > 0.5)
        if probe_indices.size == 0:
            continue
        probe_events += 1
        probe_step = int(probe_indices[0])
        pre = valid[:probe_step, agent]
        post_end = min(gates.shape[0], probe_step + window + 1)
        post = valid[probe_step + 1 : post_end, agent]
        if np.any(pre):
            pre_values.append(float(np.mean(alignments[:probe_step, agent][pre])))
        if np.any(post):
            post_values.append(float(np.mean(alignments[probe_step + 1 : post_end, agent][post])))
    pre_alignment = float(np.mean(pre_values)) if pre_values else None
    post_alignment = float(np.mean(post_values)) if post_values else None
    update = (
        post_alignment - pre_alignment
        if pre_alignment is not None and post_alignment is not None
        else None
    )
    return {
        "probe_events": probe_events,
        "agents_evaluated": min(len(pre_values), len(post_values)),
        "pre_probe_alignment": pre_alignment,
        "post_probe_alignment": post_alignment,
        "prediction_update": update,
        "censored": update is None,
    }


def probe_cost(probe_gates: np.ndarray, cost: float) -> float:
    """Return the exact accumulated probe cost for a trace."""
    if cost < 0:
        raise ValueError("cost must be non-negative")
    return float(np.sum(np.asarray(probe_gates, dtype=np.float64)) * cost)


def reversal_adaptation_latency(
    moves: np.ndarray,
    flows: np.ndarray,
    reversal_step: int,
    *,
    alive: np.ndarray | None = None,
    threshold: float = 0.0,
    consecutive: int = 2,
) -> int | None:
    """Return steps after reversal until alignment clears a stable threshold."""
    if reversal_step < 0:
        raise ValueError("reversal_step must be non-negative")
    if consecutive < 1:
        raise ValueError("consecutive must be positive")
    alignments, valid = _alignment(moves, flows, alive)
    if reversal_step >= alignments.shape[0]:
        raise ValueError("reversal_step must be inside the trace")
    mean_alignment = np.full(alignments.shape[0], np.nan, dtype=np.float64)
    for step in range(reversal_step, alignments.shape[0]):
        if np.any(valid[step]):
            mean_alignment[step] = float(np.mean(alignments[step][valid[step]]))
    for start in range(reversal_step, alignments.shape[0] - consecutive + 1):
        window = mean_alignment[start : start + consecutive]
        if np.all(np.isfinite(window)) and np.all(window >= threshold):
            return start - reversal_step
    return None


def cue_reliance_proxy(direct_alignment: float, decorrelated_alignment: float) -> float:
    """Estimate reliance on the visible flow cue."""
    return float(direct_alignment - decorrelated_alignment)


def run_frozen_causal_rollout(
    trained_state,
    config,
    *,
    steps: int = 16,
    reverse_flow: bool = False,
    flow_scale: float = 1.0,
    evaluation_seed: int | None = None,
) -> dict[str, object]:
    """Run a frozen indirect-style controller with separated trace labels."""
    if steps < 2:
        raise ValueError("steps must be at least two")
    if trained_state.embodied is None or not config.embodied.enabled:
        raise ValueError("causal rollout requires an enabled embodied controller")
    eval_config = replace(config, seed=evaluation_seed) if evaluation_seed is not None else config
    eval_config = replace(
        eval_config,
        avida=replace(eval_config.avida, enabled=False),
        world=replace(eval_config.world, drifting=False),
    )
    state = initialize_sim(eval_config)
    _copy_trained_policy(trained_state, state)
    lineage_counter = eval_config.embodied.population_size
    split = max(1, steps // 2)
    initial_flow = jnp.array(state.world.flow)
    observations: list[np.ndarray] = []
    moves: list[np.ndarray] = []
    flows: list[np.ndarray] = []
    probe_gates: list[np.ndarray] = []
    probe_signals: list[np.ndarray] = []
    alive: list[np.ndarray] = []

    for step in range(steps):
        state.world.flow = _evaluation_flow(
            initial_flow,
            step=step,
            split=split,
            reverse_flow=reverse_flow,
            flow_scale=flow_scale,
        )
        if state.embodied is None:
            raise RuntimeError("embodied state disappeared during causal rollout")
        grid = positions_to_grid(
            state.embodied.positions,
            state.world.flow.shape[0],
            state.world.flow.shape[1],
        )
        flows.append(np.asarray(state.world.flow[grid[:, 0], grid[:, 1]]))
        alive.append(np.asarray(state.embodied.alive))
        step_sim(
            state,
            eval_config,
            lineage_counter,
            allow_reproduction=False,
            observation_capture=lambda value: observations.append(np.asarray(value)),
        )
        moves.append(np.asarray(state.embodied.last_action[:, :2]))
        probe_gates.append(np.asarray(state.embodied.last_action[:, 8]))
        if state.embodied.probe_result is None:
            probe_signals.append(np.zeros((state.embodied.last_action.shape[0], 2)))
        else:
            probe_signals.append(np.asarray(state.embodied.probe_result))

    move_array = np.stack(moves)
    flow_array = np.stack(flows)
    gate_array = np.stack(probe_gates)
    alive_array = np.stack(alive)
    update = prediction_update_proxy(move_array, flow_array, gate_array, alive_array)
    cost = probe_cost(gate_array, eval_config.embodied.probe_cost)
    latency = (
        reversal_adaptation_latency(
            move_array,
            flow_array,
            split,
            alive=alive_array,
        )
        if reverse_flow
        else None
    )
    return {
        "schema_version": 1,
        "steps": steps,
        "evaluation_seed": evaluation_seed,
        "config": {
            "avida_enabled": False,
            "reverse_flow": reverse_flow,
            "flow_scale": flow_scale,
            "reversal_step": split if reverse_flow else None,
            "probe_cost": float(eval_config.embodied.probe_cost),
        },
        "agent_visible": {
            "observations": np.stack(observations).tolist(),
            "probe_signals": np.stack(probe_signals).tolist(),
        },
        "evaluator_only": {
            "local_flow": flow_array.tolist(),
            "alive": alive_array.tolist(),
            "move": move_array.tolist(),
            "probe_gate": gate_array.tolist(),
        },
        "metrics": {
            **update,
            "probe_cost": cost,
            "net_probe_value": (
                update["prediction_update"] - cost
                if update["prediction_update"] is not None
                else None
            ),
            "flow_reversal_adaptation_latency_proxy": latency,
        },
    }


__all__ = [
    "cue_reliance_proxy",
    "prediction_update_proxy",
    "probe_cost",
    "reversal_adaptation_latency",
    "run_frozen_causal_rollout",
]
