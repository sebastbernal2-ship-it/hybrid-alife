"""Bounded learned-controller pilot for the explicit causal task."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.controller import forward
from hybrid_alife.experiments.awareness_benchmark import (
    action_from_logits,
    encode_observation,
    init_learned_controller,
)
from hybrid_alife.experiments.causal_task import (
    Action,
    BenchmarkConfig,
    CausalTaskEpisode,
    WorldKind,
)


def _mutate_controller(
    params: dict[str, jax.Array], key: jax.Array, mutation_scale: float
) -> dict[str, jax.Array]:
    keys = jax.random.split(key, len(params))
    return {
        name: value + mutation_scale * jax.random.normal(subkey, value.shape)
        for (name, value), subkey in zip(sorted(params.items()), keys, strict=True)
    }


def _controller_action(
    params: dict[str, jax.Array],
    observation: dict[str, int | None | bool],
    hidden: jax.Array,
    horizon: int,
) -> tuple[Action, jax.Array]:
    encoded = jnp.asarray(encode_observation(observation, horizon=horizon))[None, :]
    logits, next_hidden = forward(params, encoded, hidden, use_memory=True)
    return action_from_logits(np.asarray(logits[0])), next_hidden


def run_causal_episode(
    kind: WorldKind | str,
    seed: int,
    params: dict[str, jax.Array],
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run one controller episode with visible and evaluator traces separated."""
    config = config or BenchmarkConfig()
    episode = CausalTaskEpisode(kind, seed, config)
    hidden_size = int(params["w_xz"].shape[-1])
    hidden = jnp.zeros((1, hidden_size), dtype=jnp.float32)
    rewards: list[float] = []
    actions: list[str] = []
    agent_trace: list[dict[str, Any]] = []
    evaluator_trace: list[dict[str, Any]] = []
    for _ in range(config.horizon):
        visible = dict(episode.agent_observation)
        action, hidden = _controller_action(params, visible, hidden, config.horizon)
        result = episode.step(action)
        rewards.append(result.reward)
        actions.append(action.value)
        agent_trace.append({"observation": visible, "action": action.value})
        evaluator_trace.append(
            {
                "hidden_rule": result.evaluator_labels["hidden_rule"],
                "correct": result.evaluator_labels["correct"],
                "informative_probe": result.evaluator_labels["informative_probe"],
            }
        )
    probe_actions = {Action.PROBE.value, Action.TARGETED_A.value, Action.TARGETED_B.value}
    probes = sum(action in probe_actions for action in actions)
    informative = sum(item["informative_probe"] for item in evaluator_trace)
    correct = [item["correct"] for item in evaluator_trace if item["correct"] is not None]
    return {
        "world": WorldKind(kind).value,
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "probe_rate": float(probes / config.horizon),
        "probe_informative_rate": float(informative / probes) if probes else 0.0,
        "correct_exploit_rate": float(np.mean(correct)) if correct else 0.0,
        "abstain_rate": float(actions.count(Action.ABSTAIN.value) / config.horizon),
        "agent_trace": agent_trace,
        "evaluator_trace": evaluator_trace,
    }


def train_causal_controller(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1, 2),
    training_worlds: tuple[WorldKind, ...] = tuple(WorldKind),
    generations: int = 2,
    population_size: int = 4,
    hidden: int = 8,
    mutation_scale: float = 0.05,
    config: BenchmarkConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Optimize bounded ladder reward without exposing evaluator labels."""
    if not training_seeds:
        raise ValueError("training_seeds must not be empty")
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be positive and population_size at least two")
    if mutation_scale <= 0:
        raise ValueError("mutation_scale must be positive")
    config = config or BenchmarkConfig()
    population = [
        init_learned_controller(seed + index, hidden=hidden)
        for index in range(population_size)
    ]
    history: list[dict[str, float | int]] = []
    for generation in range(generations):
        scores = np.asarray(
            [
                np.mean(
                    [
                        run_causal_episode(kind, episode_seed, params, config)["total_reward"]
                        for kind in training_worlds
                        for episode_seed in training_seeds
                    ]
                )
                for params in population
            ],
            dtype=np.float64,
        )
        best_index = int(np.argmax(scores))
        best = population[best_index]
        history.append(
            {
                "generation": generation,
                "best_reward": float(scores[best_index]),
                "mean_reward": float(np.mean(scores)),
            }
        )
        if generation + 1 < generations:
            keys = jax.random.split(jax.random.PRNGKey(seed + generation + 1), population_size - 1)
            population = [best] + [_mutate_controller(best, key, mutation_scale) for key in keys]
    return best, {
        "status": "pilot_only",
        "seed": seed,
        "generations": generations,
        "population_size": population_size,
        "training_seeds": list(training_seeds),
        "training_worlds": [kind.value for kind in training_worlds],
        "history": history,
    }


def _baseline_action(name: str, episode: CausalTaskEpisode, rng: np.random.Generator) -> Action:
    if name == "oracle":
        return episode.oracle_action()
    if name == "abstain":
        return Action.ABSTAIN
    if name == "random":
        return Action(rng.choice(list(Action)))
    if name == "cue":
        cue = episode.agent_observation["cue"]
        if cue is None:
            return Action.ABSTAIN
        return Action.EXPLOIT_B if cue else Action.EXPLOIT_A
    raise ValueError(f"unknown baseline: {name}")


def _run_baseline(
    name: str, kind: WorldKind, seed: int, config: BenchmarkConfig
) -> dict[str, float]:
    episode = CausalTaskEpisode(kind, seed, config)
    rng = np.random.default_rng(seed + 90_000)
    rewards: list[float] = []
    probes = 0
    correct: list[bool] = []
    for _ in range(config.horizon):
        action = _baseline_action(name, episode, rng)
        if action in (Action.PROBE, Action.TARGETED_A, Action.TARGETED_B):
            probes += 1
        result = episode.step(action)
        rewards.append(result.reward)
        if result.correct is not None:
            correct.append(result.correct)
    return {
        "mean_reward": float(np.mean(rewards)),
        "probe_rate": float(probes / config.horizon),
        "correct_exploit_rate": float(np.mean(correct)) if correct else 0.0,
    }


def evaluate_causal_controller(
    params: dict[str, jax.Array],
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Evaluate learned and shortcut-control policies across the ladder."""
    config = config or BenchmarkConfig()
    learned: dict[str, dict[str, float]] = {}
    for kind in WorldKind:
        records = [run_causal_episode(kind, seed, params, config) for seed in seeds]
        learned[kind.value] = {
            "mean_reward": float(np.mean([record["total_reward"] for record in records])),
            "probe_rate": float(np.mean([record["probe_rate"] for record in records])),
            "probe_informative_rate": float(
                np.mean([record["probe_informative_rate"] for record in records])
            ),
            "correct_exploit_rate": float(
                np.mean([record["correct_exploit_rate"] for record in records])
            ),
            "abstain_rate": float(np.mean([record["abstain_rate"] for record in records])),
        }
    baselines: dict[str, dict[str, dict[str, float]]] = {}
    for name in ("oracle", "random", "abstain", "cue"):
        baselines[name] = {
            kind.value: {
                metric: float(
                    np.mean([_run_baseline(name, kind, seed, config)[metric] for seed in seeds])
                )
                for metric in ("mean_reward", "probe_rate", "correct_exploit_rate")
            }
            for kind in WorldKind
        }
    return {"status": "pilot_only", "learned": learned, "baselines": baselines}


__all__ = [
    "evaluate_causal_controller",
    "run_causal_episode",
    "train_causal_controller",
]
