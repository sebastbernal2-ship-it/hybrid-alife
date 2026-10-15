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


def _training_score(record: dict[str, Any], objective: str) -> float:
    if objective == "reward":
        return float(record["total_reward"])
    if objective == "inquiry_shaped":
        redundant = record["probe_count"] - record["informative_probe_count"]
        return float(
            record["total_reward"]
            + 0.5 * record["informative_probe_count"]
            - 0.5 * redundant
            + 0.5 * record["post_probe_correct_exploits"]
        )
    if objective == "value_of_information":
        gain = float(record["post_probe_return_gain"])
        redundant = max(record["probe_count"] - 1, 0)
        return float(
            record["total_reward"]
            + 0.5 * max(gain, 0.0)
            - 0.25 * max(-gain, 0.0)
            - 0.25 * redundant
        )
    raise ValueError(f"unknown training objective: {objective}")


def _collect_policy_episode(
    params: dict[str, jax.Array],
    kind: WorldKind,
    seed: int,
    config: BenchmarkConfig,
    key: jax.Array,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    episode = CausalTaskEpisode(kind, seed, config)
    hidden_size = int(params["w_xz"].shape[-1])
    hidden = jnp.zeros((1, hidden_size), dtype=jnp.float32)
    observations: list[np.ndarray] = []
    action_indices: list[int] = []
    rewards: list[float] = []
    action_order = tuple(Action)
    for _ in range(config.horizon):
        visible = episode.agent_observation
        encoded = encode_observation(visible, horizon=config.horizon)
        raw, hidden = forward(params, jnp.asarray(encoded)[None, :], hidden, use_memory=True)
        key, sample_key = jax.random.split(key)
        index = int(jax.random.categorical(sample_key, raw[0]))
        result = episode.step(action_order[index])
        observations.append(encoded)
        action_indices.append(index)
        rewards.append(result.reward)
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        returns[index] = running
    return (
        np.asarray(observations, dtype=np.float32),
        np.asarray(action_indices, dtype=np.int32),
        returns,
        float(sum(rewards)),
    )


def _policy_gradient_loss(
    params: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    returns: jax.Array,
    entropy_coef: float,
) -> jax.Array:
    hidden_size = params["w_xz"].shape[-1]
    hidden = jnp.zeros((1, hidden_size), dtype=jnp.float32)
    log_probs: list[jax.Array] = []
    entropies: list[jax.Array] = []
    for index in range(observations.shape[0]):
        raw, hidden = forward(
            params,
            observations[index][None, :],
            hidden,
            use_memory=True,
        )
        logits = raw[0]
        log_prob = jax.nn.log_softmax(logits)
        probabilities = jax.nn.softmax(logits)
        log_probs.append(log_prob[actions[index]])
        entropies.append(-jnp.sum(probabilities * log_prob))
    return -jnp.mean(jnp.asarray(log_probs) * returns) - entropy_coef * jnp.mean(
        jnp.asarray(entropies)
    )


def train_policy_gradient_controller(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1, 2),
    training_worlds: tuple[WorldKind, ...] = tuple(WorldKind),
    updates: int = 8,
    episodes_per_update: int = 12,
    hidden: int = 8,
    learning_rate: float = 0.01,
    entropy_coef: float = 0.01,
    gamma: float = 1.0,
    config: BenchmarkConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Train with REINFORCE and entropy regularization using task returns only."""
    if not training_seeds:
        raise ValueError("training_seeds must not be empty")
    if not training_worlds:
        raise ValueError("training_worlds must not be empty")
    if updates < 1 or episodes_per_update < 1:
        raise ValueError("updates and episodes_per_update must be positive")
    if learning_rate <= 0 or entropy_coef < 0 or not 0 < gamma <= 1:
        raise ValueError("invalid policy-gradient hyperparameters")
    config = config or BenchmarkConfig()
    params = init_learned_controller(seed, hidden=hidden)
    specs = [
        (kind, episode_seed)
        for kind in training_worlds
        for episode_seed in training_seeds
    ]
    master_key = jax.random.PRNGKey(seed + 70_000)
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(master_key, episodes_per_update + 1)
        master_key = keys[0]
        batches = [
            _collect_policy_episode(
                params,
                *specs[index % len(specs)],
                config,
                keys[index + 1],
                gamma,
            )
            for index in range(episodes_per_update)
        ]
        observations = np.concatenate([batch[0] for batch in batches])
        actions = np.concatenate([batch[1] for batch in batches])
        returns = np.concatenate([batch[2] for batch in batches])
        returns = (returns - returns.mean()) / (returns.std() + 1e-6)
        loss, gradients = jax.value_and_grad(_policy_gradient_loss)(
            params,
            jnp.asarray(observations),
            jnp.asarray(actions),
            jnp.asarray(returns),
            entropy_coef,
        )
        params = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * jnp.clip(gradient, -1.0, 1.0),
            params,
            gradients,
        )
        history.append(
            {
                "update": update,
                "loss": float(loss),
                "mean_return": float(np.mean([batch[3] for batch in batches])),
            }
        )
    return params, {
        "status": "pilot_only",
        "trainer": "policy_gradient",
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "learning_rate": learning_rate,
        "entropy_coef": entropy_coef,
        "gamma": gamma,
        "training_seeds": list(training_seeds),
        "training_worlds": [kind.value for kind in training_worlds],
        "history": history,
    }


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
    first_probe = next(
        (index for index, action in enumerate(actions) if action in probe_actions),
        None,
    )
    if first_probe is None:
        post_probe_return_gain = 0.0
    else:
        pre_rewards = rewards[:first_probe]
        post_end = next(
            (
                index
                for index in range(first_probe + 1, len(actions))
                if actions[index] in probe_actions
            ),
            len(actions),
        )
        post_rewards = rewards[first_probe + 1 : post_end]
        pre_mean = float(np.mean(pre_rewards)) if pre_rewards else 0.0
        post_mean = float(np.mean(post_rewards)) if post_rewards else 0.0
        post_probe_return_gain = post_mean - pre_mean
    informative_seen = False
    post_probe_exploits = 0
    post_probe_correct_exploits = 0
    for action, label in zip(actions, evaluator_trace, strict=True):
        if label["informative_probe"]:
            informative_seen = True
        if informative_seen and action in (Action.EXPLOIT_A.value, Action.EXPLOIT_B.value):
            post_probe_exploits += 1
            post_probe_correct_exploits += int(label["correct"] == 1)
    return {
        "world": WorldKind(kind).value,
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "probe_count": probes,
        "informative_probe_count": informative,
        "probe_rate": float(probes / config.horizon),
        "probe_informative_rate": float(informative / probes) if probes else 0.0,
        "post_probe_exploits": post_probe_exploits,
        "post_probe_correct_exploits": post_probe_correct_exploits,
        "post_probe_return_gain": post_probe_return_gain,
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
    objective: str = "reward",
    config: BenchmarkConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Optimize bounded ladder reward without exposing evaluator labels."""
    if not training_seeds:
        raise ValueError("training_seeds must not be empty")
    if generations < 1 or population_size < 2:
        raise ValueError("generations must be positive and population_size at least two")
    if mutation_scale <= 0:
        raise ValueError("mutation_scale must be positive")
    if objective not in {"reward", "inquiry_shaped", "value_of_information"}:
        raise ValueError(f"unknown training objective: {objective}")
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
                        _training_score(
                            run_causal_episode(kind, episode_seed, params, config),
                            objective,
                        )
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
        "objective": objective,
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
    "train_policy_gradient_controller",
]
