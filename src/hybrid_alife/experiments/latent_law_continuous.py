"""Continuous-thrust policy for the latent gravity-law task.

A discrete action set makes the policy learn a broad mixed strategy whose mode
is arbitrary. A continuous thrust has a single mean, so deterministic
deployment is the mean and there is no mode-versus-mean gap.

The policy is a linear Gaussian over the encoded observation. Because the
observation carries the organism's own recent velocity and thrust, gravity is a
linear function of it, so a linear mean can express the optimal policy exactly.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_controller import (
    OBS_DIM,
    encode_gravity_observation,
)
from hybrid_alife.experiments.latent_law_task import GravityConfig, GravityEpisode, Treatment


def init_continuous_policy(
    seed: int, *, obs_dim: int = OBS_DIM
) -> dict[str, jax.Array]:
    """Linear Gaussian thrust policy: mean ``w @ obs + b`` with fixed noise."""
    keys = jax.random.split(jax.random.PRNGKey(seed + 4321), 2)
    return {
        "w": 0.01 * jax.random.normal(keys[0], (obs_dim, 2)),
        "b": jnp.zeros((2,)),
    }


def continuous_mean(
    policy: dict[str, jax.Array], encoded: jax.Array
) -> jax.Array:
    """Unbounded mean thrust for one observation."""
    return encoded @ policy["w"] + policy["b"]


def _clip_thrust(thrust: jax.Array) -> jax.Array:
    norm = jnp.linalg.norm(thrust)
    return jnp.where(norm > 1.0, thrust / jnp.maximum(norm, 1e-8), thrust)


def run_continuous_episode(
    params: dict[str, jax.Array],
    seed: int,
    *,
    treatment: Treatment = Treatment.NO_REVELATION,
    config: GravityConfig | None = None,
    deterministic: bool = True,
    exploration_std: float = 0.4,
    key: jax.Array | None = None,
) -> dict[str, Any]:
    """Run one episode, deploying the mean when ``deterministic`` is true."""
    config = config or GravityConfig()
    episode = GravityEpisode(seed, treatment, config)
    if key is None:
        key = jax.random.PRNGKey(seed + 55_000)
    thrust_trace: list[tuple[float, float]] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        encoded = encode_gravity_observation(episode.agent_observation, config.horizon)
        thrust = continuous_mean(params, jnp.asarray(encoded))
        if not deterministic:
            key, sample_key = jax.random.split(key)
            noise = exploration_std * jax.random.normal(sample_key, thrust.shape)
            thrust = thrust + noise
        clipped = np.asarray(_clip_thrust(thrust))
        result = episode.step(clipped)
        thrust_trace.append((float(clipped[0]), float(clipped[1])))
        rewards.append(result.reward)
    return {
        "seed": seed,
        "treatment": Treatment(treatment).value,
        "total_reward": float(sum(rewards)),
        "thrust_trace": thrust_trace,
        "mean_gravity_alignment": float(
            np.mean(
                [
                    float(np.dot(np.asarray(t), np.asarray(episode.gravity_vector))) / 0.2
                    for t in thrust_trace
                ]
            )
        ),
    }


def _continuous_log_prob(
    policy: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    exploration_std: float,
) -> jax.Array:
    mean = observations @ policy["w"] + policy["b"]
    residual = (actions - mean) / exploration_std
    return -0.5 * jnp.sum(residual**2, axis=-1) - jnp.log(exploration_std)


def _continuous_loss(
    policy: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    exploration_std: float,
) -> jax.Array:
    log_probs = jax.vmap(_continuous_log_prob, in_axes=(None, 0, 0, None))(
        policy, observations, actions, exploration_std
    )
    return -jnp.mean(log_probs * advantages)


def _collect_continuous_episode(
    policy: dict[str, jax.Array],
    seed: int,
    treatment: Treatment,
    config: GravityConfig,
    key: jax.Array,
    exploration_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    episode = GravityEpisode(seed, treatment, config)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        encoded = encode_gravity_observation(episode.agent_observation, config.horizon)
        mean = continuous_mean(policy, jnp.asarray(encoded))
        key, sample_key = jax.random.split(key)
        thrust = mean + exploration_std * jax.random.normal(sample_key, mean.shape)
        clipped = _clip_thrust(thrust)
        result = episode.step(np.asarray(clipped))
        observations.append(encoded)
        actions.append(np.asarray(clipped))
        rewards.append(result.reward)
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + running
        returns[index] = running
    return (
        np.asarray(observations),
        np.asarray(actions),
        returns,
        float(sum(rewards)),
    )


def train_continuous_policy(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1, 2, 3),
    training_treatments: tuple[Treatment, ...] = (Treatment.NO_REVELATION,),
    updates: int = 200,
    episodes_per_update: int = 32,
    learning_rate: float = 0.1,
    exploration_std: float = 0.4,
    config: GravityConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """REINFORCE over a linear Gaussian thrust policy."""
    if not training_seeds or not training_treatments:
        raise ValueError("training seeds and treatments must not be empty")
    if updates < 1 or episodes_per_update < 2:
        raise ValueError("updates must be positive and episodes_per_update at least two")
    if learning_rate <= 0 or exploration_std <= 0:
        raise ValueError("learning_rate and exploration_std must be positive")
    config = config or GravityConfig()
    policy = init_continuous_policy(seed)
    specs = [
        (seed_value, treatment)
        for treatment in training_treatments
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 66_000)
    loss_and_grad = jax.jit(jax.value_and_grad(_continuous_loss))
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(key, episodes_per_update + 1)
        key = keys[0]
        batches = [
            _collect_continuous_episode(
                policy,
                *specs[index % len(specs)],
                config,
                keys[index + 1],
                exploration_std,
            )
            for index in range(episodes_per_update)
        ]
        observations = np.stack([batch[0] for batch in batches])
        actions = np.stack([batch[1] for batch in batches])
        return_matrix = np.stack([batch[2] for batch in batches])
        advantages = return_matrix - return_matrix.mean(axis=0, keepdims=True)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        loss, gradients = loss_and_grad(
            policy,
            jnp.asarray(observations),
            jnp.asarray(actions),
            jnp.asarray(advantages),
            exploration_std,
        )
        policy = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * jnp.clip(gradient, -1.0, 1.0),
            policy,
            gradients,
        )
        history.append(
            {
                "update": update,
                "loss": float(loss),
                "mean_return": float(np.mean([batch[3] for batch in batches])),
            }
        )
    return policy, {
        "status": "calibration_only",
        "trainer": "continuous_policy_gradient",
        "jax_loss_jitted": True,
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "learning_rate": learning_rate,
        "exploration_std": exploration_std,
        "training_seeds": list(training_seeds),
        "training_treatments": [treatment.value for treatment in training_treatments],
        "history": history,
    }


def _collect_continuous_episode_arrays(
    policy: dict[str, jax.Array],
    seed: int,
    treatment: Treatment,
    config: GravityConfig,
    key: jax.Array,
    exploration_std: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    observations, actions, _returns, total = _collect_continuous_episode(
        policy, seed, treatment, config, key, exploration_std
    )
    return observations, actions, total


def train_continuous_policy_rwr(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1, 2, 3),
    training_treatments: tuple[Treatment, ...] = (Treatment.NO_REVELATION,),
    rounds: int = 6,
    episodes_per_round: int = 64,
    exploration_std: float = 0.5,
    exploration_std_final: float | None = None,
    temperature: float = 2.0,
    ridge: float = 1e-3,
    config: GravityConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Reward-weighted regression over a linear Gaussian thrust policy.

    The mean is linear, so the weight update is a closed-form weighted
    least-squares solve. High-return episodes dominate the fit, which makes
    the learner far more stable than a policy gradient on this task.
    """
    if not training_seeds or not training_treatments:
        raise ValueError("training seeds and treatments must not be empty")
    if rounds < 1 or episodes_per_round < 2:
        raise ValueError("rounds must be positive and episodes_per_round at least two")
    if exploration_std <= 0 or temperature <= 0 or ridge < 0:
        raise ValueError("invalid reward-weighted regression parameters")
    if exploration_std_final is not None and not 0 < exploration_std_final <= exploration_std:
        raise ValueError("exploration_std_final must be positive and at most exploration_std")
    config = config or GravityConfig()
    policy = init_continuous_policy(seed)
    specs = [
        (seed_value, treatment)
        for treatment in training_treatments
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 77_000)
    history: list[dict[str, float | int]] = []
    for round_index in range(rounds):
        # Anneal exploration so the fit sharpens once coverage is established.
        progress = round_index / max(rounds - 1, 1)
        current_std = (
            exploration_std
            if exploration_std_final is None
            else exploration_std + progress * (exploration_std_final - exploration_std)
        )
        keys = jax.random.split(key, episodes_per_round + 1)
        key = keys[0]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        totals: list[float] = []
        for index in range(episodes_per_round):
            obs, act, total = _collect_continuous_episode_arrays(
                policy,
                *specs[index % len(specs)],
                config,
                keys[index + 1],
                current_std,
            )
            observations.append(obs)
            actions.append(act)
            totals.append(total)
        stacked_observations = np.concatenate(observations, axis=0)
        stacked_actions = np.concatenate(actions, axis=0)
        improved = np.asarray(totals) - np.max(totals)
        weights = np.exp(improved / temperature)
        # Repeat each episode's weight across its steps.
        steps = observations[0].shape[0]
        step_weights = np.repeat(weights, steps)
        policy = _weighted_least_squares(
            stacked_observations, stacked_actions, step_weights, ridge
        )
        history.append(
            {
                "round": round_index,
                "best_return": float(np.max(totals)),
                "mean_return": float(np.mean(totals)),
            }
        )
    return policy, {
        "status": "calibration_only",
        "trainer": "reward_weighted_regression",
        "seed": seed,
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "exploration_std": exploration_std,
        "exploration_std_final": exploration_std_final,
        "temperature": temperature,
        "training_seeds": list(training_seeds),
        "training_treatments": [treatment.value for treatment in training_treatments],
        "history": history,
    }


def _weighted_least_squares(
    observations: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    ridge: float,
) -> dict[str, jax.Array]:
    """Closed-form weighted ridge fit of ``mean = w @ obs + b``."""
    design = np.concatenate(
        [observations, np.ones((observations.shape[0], 1))], axis=1
    ).astype(np.float64)
    weighted = design * weights[:, None]
    gram = weighted.T @ design + ridge * np.eye(design.shape[1])
    rhs = weighted.T @ actions.astype(np.float64)
    theta = np.linalg.solve(gram, rhs)
    return {
        "w": jnp.asarray(theta[:-1], dtype=jnp.float32),
        "b": jnp.asarray(theta[-1], dtype=jnp.float32),
    }


def per_direction_continuous_rewards(
    params: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    horizon: int = 12,
    action_acceleration: float = 0.25,
    treatment: Treatment = Treatment.NO_REVELATION,
) -> dict[str, float]:
    """Deterministic (mean) deployment reward, reported per gravity direction."""
    scores: dict[str, float] = {}
    for direction in directions:
        config = GravityConfig(
            horizon=horizon,
            gravity_choices=(direction,),
            action_acceleration=action_acceleration,
        )
        values = [
            run_continuous_episode(params, seed, treatment=treatment, config=config)[
                "total_reward"
            ]
            for seed in seeds
        ]
        scores[str(direction)] = float(np.mean(values))
    return scores


__all__ = [
    "continuous_mean",
    "init_continuous_policy",
    "per_direction_continuous_rewards",
    "run_continuous_episode",
    "train_continuous_policy",
    "train_continuous_policy_rwr",
]
