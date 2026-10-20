"""Internal law representation for the simulator gravity task.

Option 2 of the research alignment: the organism forms an **explicit internal
estimate** of the imposed law from its own sensors, then learns to act on that
estimate.

The estimate is legitimate self-inference, not evaluator leakage. It uses only
the organism's own last velocity, previous velocity, and previous thrust, which
are all proprioceptive quantities the organism measured itself:

```text
law_estimate = velocity - friction * previous_velocity - acceleration * previous_thrust
```

The policy maps that two-dimensional estimate to a thrust direction. The optimal
map is the identity, so learning is well posed and per-direction reliability
should no longer depend on the joint fit of every sensor coefficient.

The inference rule is a declared fixed mechanism, like an efference-copy forward
model. The *policy* on top of it is what is learned. That is the claim: the
organism forms an internal estimate of the law and learns to exploit it.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_simulator import SimulatorOrganism
from hybrid_alife.types import WorldConfig

FRICTION = 0.9
ACCELERATION = 0.25


def infer_law(
    observation: dict[str, Any],
    *,
    friction: float = FRICTION,
    acceleration: float = ACCELERATION,
) -> np.ndarray:
    """Estimate the imposed law from proprioception alone."""
    velocity = np.asarray(observation["velocity"], dtype=np.float64)
    previous_velocity = np.asarray(observation["previous_velocity"], dtype=np.float64)
    previous_thrust = np.asarray(observation["previous_thrust"], dtype=np.float64)
    return velocity - friction * previous_velocity - acceleration * previous_thrust


REVELATION_INPUT_DIM = 5


def init_estimate_policy(
    seed: int, *, input_dim: int = 2
) -> dict[str, jax.Array]:
    """Linear map from the internal estimate (or estimate plus revelation)."""
    if input_dim not in (2, REVELATION_INPUT_DIM):
        raise ValueError("input_dim must be 2 or 5")
    key = jax.random.PRNGKey(seed + 2468)
    return {
        "w": 0.01 * jax.random.normal(key, (input_dim, 2)),
        "b": jnp.zeros((2,)),
    }


def encode_policy_input(
    observation: dict[str, Any], *, use_revelation: bool
) -> np.ndarray:
    """Policy input: the internal estimate, plus the revelation when enabled."""
    estimate = infer_law(observation)
    if not use_revelation:
        return estimate
    payload = observation.get("revelation")
    present = observation.get("revelation_present", False)
    revelation = np.asarray(payload, dtype=np.float64) if present else np.zeros(2)
    return np.concatenate([estimate, [float(present)], revelation])


def estimate_thrust(
    policy: dict[str, jax.Array], policy_input: np.ndarray
) -> np.ndarray:
    """Thrust produced by the learned map applied to the policy input."""
    value = jnp.asarray(policy_input, dtype=jnp.float32) @ policy["w"] + policy["b"]
    return np.asarray(value)


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def run_inference_episode(
    policy: dict[str, jax.Array],
    direction: tuple[float, float],
    *,
    seed: int = 0,
    horizon: int = 12,
    config: WorldConfig | None = None,
    use_revelation: bool = False,
    revelation: tuple[float, float] | None = None,
    revelation_step: int = 0,
    revelation_noise: float = 0.0,
    revelation_kind: str = "exact",
) -> dict[str, Any]:
    """Deterministic deployment of the estimate-driven policy."""
    organism = SimulatorOrganism(
        direction,
        horizon=horizon,
        seed=seed,
        config=config,
        revelation=revelation,
        revelation_step=revelation_step,
        revelation_noise=revelation_noise,
        revelation_kind=revelation_kind,
    )
    rewards: list[float] = []
    estimates: list[np.ndarray] = []
    for _ in range(horizon):
        observation = organism.agent_observation
        policy_input = encode_policy_input(observation, use_revelation=use_revelation)
        estimates.append(infer_law(observation))
        thrust = _unit(estimate_thrust(policy, policy_input))
        rewards.append(organism.step(thrust))
    return {
        "direction": direction,
        "total_reward": float(sum(rewards)),
        "rewards": rewards,
        "estimates": estimates,
    }


def _collect_inference_episode(
    policy: dict[str, jax.Array],
    direction: tuple[float, float],
    seed: int,
    key: jax.Array,
    *,
    horizon: int,
    exploration_std: float,
    config: WorldConfig | None = None,
    use_revelation: bool = False,
    revelation: tuple[float, float] | None = None,
    revelation_step: int = 0,
    revelation_noise: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    organism = SimulatorOrganism(
        direction,
        horizon=horizon,
        seed=seed,
        config=config,
        revelation=revelation,
        revelation_step=revelation_step,
        revelation_noise=revelation_noise,
    )
    estimates: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(horizon):
        observation = organism.agent_observation
        policy_input = encode_policy_input(observation, use_revelation=use_revelation)
        key, sample_key = jax.random.split(key)
        noise = exploration_std * jax.random.normal(sample_key, (2,))
        thrust = _unit(estimate_thrust(policy, policy_input) + np.asarray(noise))
        estimates.append(policy_input)
        actions.append(thrust)
        rewards.append(organism.step(thrust))
    return np.asarray(estimates), np.asarray(actions), float(sum(rewards))


def fit_estimate_policy(
    estimates: np.ndarray,
    actions: np.ndarray,
    weights: np.ndarray,
    ridge: float = 1e-3,
) -> dict[str, jax.Array]:
    """Closed-form weighted ridge fit of ``thrust = estimate @ w + b``."""
    design = np.concatenate([estimates, np.ones((estimates.shape[0], 1))], axis=1)
    design = design.astype(np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    n_features = estimates.shape[1]
    total = float(weights.sum())
    mean = (design * weights[:, None]).sum(axis=0) / max(total, 1e-12)
    variance = ((design - mean) ** 2 * weights[:, None]).sum(axis=0) / max(total, 1e-12)
    scale = np.sqrt(variance)
    scale[scale < 1e-6] = 1.0
    mean[n_features] = 0.0
    scale[n_features] = 1.0
    scaled = (design - mean) / scale
    weighted = scaled * weights[:, None]
    gram = weighted.T @ scaled + ridge * np.eye(scaled.shape[1])
    rhs = weighted.T @ actions.astype(np.float64)
    theta = np.linalg.solve(gram, rhs)
    w = theta[:-1] / scale[:-1, None]
    b = theta[-1] - np.sum(theta[:-1] * (mean[:-1] / scale[:-1])[:, None], axis=0)
    return {
        "w": jnp.asarray(w, dtype=jnp.float32),
        "b": jnp.asarray(b, dtype=jnp.float32),
    }


def train_estimate_policy_rwr(
    seed: int,
    *,
    training_directions: tuple[tuple[float, float], ...],
    training_seeds: tuple[int, ...] = (0, 1, 2, 3),
    rounds: int = 12,
    episodes_per_round: int = 64,
    horizon: int = 12,
    exploration_std: float = 0.4,
    exploration_std_final: float = 0.05,
    temperature: float = 2.0,
    ridge: float = 1e-3,
    config: WorldConfig | None = None,
    use_revelation: bool = False,
    train_with_revelation: bool = False,
    revelation_noise: float = 0.0,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Reward-weighted regression over the internal-estimate policy.

    With ``use_revelation`` the policy input also carries the revelation and a
    presence flag. ``train_with_revelation`` delivers a true revelation at step
    zero during training.
    """
    if not training_directions or not training_seeds:
        raise ValueError("training directions and seeds must not be empty")
    if rounds < 1 or episodes_per_round < 1:
        raise ValueError("rounds and episodes_per_round must be positive")
    policy = init_estimate_policy(
        seed, input_dim=REVELATION_INPUT_DIM if use_revelation else 2
    )
    specs = [
        (direction, seed_value)
        for direction in training_directions
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 99_000)
    history: list[dict[str, float]] = []
    for round_index in range(rounds):
        progress = round_index / max(rounds - 1, 1)
        current_std = exploration_std + progress * (
            exploration_std_final - exploration_std
        )
        keys = jax.random.split(key, episodes_per_round + 1)
        key = keys[0]
        all_estimates: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        totals: list[float] = []
        for index in range(episodes_per_round):
            direction, seed_value = specs[index % len(specs)]
            revelation = direction if (use_revelation and train_with_revelation) else None
            estimates, actions, total = _collect_inference_episode(
                policy,
                direction,
                seed_value,
                keys[index + 1],
                horizon=horizon,
                exploration_std=current_std,
                config=config,
                use_revelation=use_revelation,
                revelation=revelation,
                revelation_noise=revelation_noise,
            )
            all_estimates.append(estimates)
            all_actions.append(actions)
            totals.append(total)
        stacked_estimates = np.concatenate(all_estimates, axis=0)
        stacked_actions = np.concatenate(all_actions, axis=0)
        weights = np.exp((np.asarray(totals) - np.max(totals)) / temperature)
        step_weights = np.repeat(weights, horizon)
        policy = fit_estimate_policy(
            stacked_estimates, stacked_actions, step_weights, ridge
        )
        history.append(
            {
                "round": float(round_index),
                "best_return": float(np.max(totals)),
                "mean_return": float(np.mean(totals)),
            }
        )
    return policy, {
        "status": "calibration_only",
        "trainer": "internal_estimate_reward_weighted_regression",
        "seed": seed,
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "horizon": horizon,
        "exploration_std": exploration_std,
        "exploration_std_final": exploration_std_final,
        "temperature": temperature,
        "use_revelation": use_revelation,
        "train_with_revelation": train_with_revelation,
        "training_directions": [list(direction) for direction in training_directions],
        "history": history,
    }


def per_direction_inference_rewards(
    policy: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    horizon: int = 12,
    config: WorldConfig | None = None,
    use_revelation: bool = False,
    reveal: bool = False,
) -> dict[str, float]:
    """Deterministic reward per imposed-flow direction, via the internal estimate.

    With ``reveal`` the true law is delivered at step zero.
    """
    scores: dict[str, float] = {}
    for direction in directions:
        values = [
            run_inference_episode(
                policy,
                direction,
                seed=seed,
                horizon=horizon,
                config=config,
                use_revelation=use_revelation,
                revelation=direction if reveal else None,
            )["total_reward"]
            for seed in seeds
        ]
        scores[str(direction)] = float(np.mean(values))
    return scores


__all__ = [
    "REVELATION_INPUT_DIM",
    "encode_policy_input",
    "fit_estimate_policy",
    "infer_law",
    "init_estimate_policy",
    "per_direction_inference_rewards",
    "run_inference_episode",
    "train_estimate_policy_rwr",
]
