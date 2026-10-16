"""Calibration controller for the latent gravity-law task."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.controller import forward, init_controller_params
from hybrid_alife.experiments.latent_law_task import (
    _ACTION_VECTORS,
    Action,
    GravityConfig,
    GravityEpisode,
    Treatment,
)

OBS_DIM = 13
_ACTIONS = tuple(Action)


def encode_gravity_observation(observation: dict[str, Any], horizon: int) -> np.ndarray:
    revelation = observation["revelation"] or {}
    channel = revelation.get("channel")
    content = revelation.get("content")
    content_x = 0.0
    content_y = 0.0
    if isinstance(content, tuple) and len(content) == 2:
        content_x, content_y = content
    elif isinstance(content, dict) and "velocity_change" in content:
        content_x, content_y = content["velocity_change"]
    return np.asarray(
        [
            observation["step"] / horizon,
            *observation["position"],
            *observation["velocity"],
            observation["surface"],
            float(observation["revelation_present"]),
            float(channel == "visual_demonstration"),
            float(channel == "noisy_sensor"),
            float(channel == "external_message"),
            float(content_x),
            float(content_y),
            float(revelation.get("reliability", 0.0)),
        ],
        dtype=np.float32,
    )


def _action_from_logits(logits: np.ndarray) -> Action:
    return _ACTIONS[int(np.argmax(logits[: len(_ACTIONS)]))]


def run_gravity_episode(
    seed: int,
    treatment: Treatment | str,
    params: dict[str, jax.Array],
    config: GravityConfig | None = None,
) -> dict[str, Any]:
    config = config or GravityConfig()
    episode = GravityEpisode(seed, treatment, config)
    state = _policy_state(params)
    agent_trace: list[dict[str, Any]] = []
    evaluator_trace: list[dict[str, Any]] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        observation = dict(episode.agent_observation)
        encoded = encode_gravity_observation(observation, config.horizon)
        logits, state = _policy_logits(params, encoded, state)
        action = _action_from_logits(logits)
        result = episode.step(action)
        rewards.append(result.reward)
        agent_trace.append({"observation": observation, "action": action.value})
        gravity = np.asarray(result.evaluator_labels["gravity_vector"])
        move = _ACTION_VECTORS[action]
        alignment = float(np.dot(move, gravity) / max(np.linalg.norm(gravity), 1e-8))
        evaluator_trace.append(
            {
                **result.evaluator_labels,
                "action": action.value,
                "gravity_alignment": alignment,
            }
        )
    revelation_steps = [
        index for index, item in enumerate(agent_trace) if item["observation"]["revelation_present"]
    ]
    after_revelation = [
        item["action"] != Action.REST.value
        for item in agent_trace[revelation_steps[0] + 1 :]
    ] if revelation_steps else []
    return {
        "status": "calibration_only",
        "treatment": Treatment(treatment).value,
        "seed": seed,
        "total_reward": float(sum(rewards)),
        "mean_gravity_alignment": float(
            np.mean([item["gravity_alignment"] for item in evaluator_trace])
        ),
        "post_revelation_action_rate": float(np.mean(after_revelation))
        if after_revelation
        else 0.0,
        "agent_trace": agent_trace,
        "evaluator_trace": evaluator_trace,
        "revelation_log": episode.revelation_log,
    }


def is_linear_policy(policy: dict[str, jax.Array]) -> bool:
    """A linear policy has ``w`` and ``b``; a recurrent policy has ``w_xz``."""
    return "w" in policy and "w_xz" not in policy


def init_linear_policy(seed: int, *, obs_dim: int = OBS_DIM) -> dict[str, jax.Array]:
    """A single linear layer over the encoded observation.

    The encoding contains ``step / horizon``, so a linear policy can still
    express a time-dependent action schedule without recurrence.
    """
    keys = jax.random.split(jax.random.PRNGKey(seed + 1234), 2)
    return {
        "w": 0.01 * jax.random.normal(keys[0], (obs_dim, len(_ACTIONS))),
        "b": jnp.zeros((len(_ACTIONS),)),
    }


def _policy_state(policy: dict[str, jax.Array]) -> jax.Array | None:
    if is_linear_policy(policy):
        return None
    return jnp.zeros((1, int(policy["w_xz"].shape[-1])), dtype=jnp.float32)


def _policy_logits(
    policy: dict[str, jax.Array],
    encoded: np.ndarray,
    state: jax.Array | None,
) -> tuple[np.ndarray, jax.Array | None]:
    """Return action logits and the next policy state for either policy type."""
    if is_linear_policy(policy):
        logits = jnp.asarray(encoded) @ policy["w"] + policy["b"]
        return np.asarray(logits), state
    raw, next_state = forward(
        policy, jnp.asarray(encoded)[None, :], state, use_memory=True
    )
    return np.asarray(raw[0]), next_state


def _linear_episode_log_probs_and_entropy(
    policy: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    logits = observations @ policy["w"] + policy["b"]
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jax.nn.softmax(logits, axis=-1)
    entropy = -jnp.sum(probs * log_probs, axis=-1)
    return log_probs[jnp.arange(actions.shape[0]), actions], entropy


def _linear_loss(
    policy: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
    entropy_coef: float,
) -> jax.Array:
    log_probs, entropies = jax.vmap(
        _linear_episode_log_probs_and_entropy, in_axes=(None, 0, 0)
    )(policy, observations, actions)
    return -jnp.mean(log_probs * advantages) - entropy_coef * jnp.mean(entropies)


def _collect_linear_episode(
    policy: dict[str, jax.Array],
    seed: int,
    treatment: Treatment,
    config: GravityConfig,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    episode = GravityEpisode(seed, treatment, config)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        encoded = encode_gravity_observation(episode.agent_observation, config.horizon)
        logits = jnp.asarray(encoded) @ policy["w"] + policy["b"]
        key, sample_key = jax.random.split(key)
        index = int(jax.random.categorical(sample_key, logits))
        result = episode.step(_ACTIONS[index])
        observations.append(encoded)
        actions.append(index)
        rewards.append(result.reward)
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + running
        returns[index] = running
    return np.asarray(observations), np.asarray(actions), returns, float(sum(rewards))


def train_linear_gravity_policy(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1),
    training_treatments: tuple[Treatment, ...] = (Treatment.NO_REVELATION,),
    updates: int = 200,
    episodes_per_update: int = 32,
    learning_rate: float = 0.05,
    entropy_coef: float = 0.02,
    config: GravityConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Plain REINFORCE with a per-timestep mean baseline and entropy control."""
    if not training_seeds or not training_treatments:
        raise ValueError("training seeds and treatments must not be empty")
    if updates < 1 or episodes_per_update < 2:
        raise ValueError("updates must be positive and episodes_per_update at least two")
    if learning_rate <= 0 or entropy_coef < 0:
        raise ValueError("invalid linear policy hyperparameters")
    config = config or GravityConfig()
    policy = init_linear_policy(seed)
    specs = [
        (seed_value, treatment)
        for treatment in training_treatments
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 80_000)
    loss_and_grad = jax.jit(jax.value_and_grad(_linear_loss))
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(key, episodes_per_update + 1)
        key = keys[0]
        batches = [
            _collect_linear_episode(
                policy, *specs[index % len(specs)], config, keys[index + 1]
            )
            for index in range(episodes_per_update)
        ]
        observations = np.stack([batch[0] for batch in batches])
        actions = np.stack([batch[1] for batch in batches])
        return_matrix = np.stack([batch[2] for batch in batches])
        # Per-timestep mean baseline, then normalise so the step size is stable.
        advantages = return_matrix - return_matrix.mean(axis=0, keepdims=True)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        loss, gradients = loss_and_grad(
            policy,
            jnp.asarray(observations),
            jnp.asarray(actions),
            jnp.asarray(advantages),
            entropy_coef,
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
        "trainer": "linear_policy_gradient",
        "jax_loss_jitted": True,
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "entropy_coef": entropy_coef,
        "learning_rate": learning_rate,
        "training_seeds": list(training_seeds),
        "training_treatments": [treatment.value for treatment in training_treatments],
        "history": history,
    }


def _collect_episode(
    params: dict[str, jax.Array],
    seed: int,
    treatment: Treatment,
    config: GravityConfig,
    key: jax.Array,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    episode = GravityEpisode(seed, treatment, config)
    hidden = jnp.zeros((1, int(params["w_xz"].shape[-1])), dtype=jnp.float32)
    observations: list[np.ndarray] = []
    actions: list[int] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        encoded = encode_gravity_observation(episode.agent_observation, config.horizon)
        raw, hidden = forward(params, jnp.asarray(encoded)[None, :], hidden, use_memory=True)
        key, sample_key = jax.random.split(key)
        action_index = int(jax.random.categorical(sample_key, raw[0]))
        result = episode.step(_ACTIONS[action_index])
        observations.append(encoded)
        actions.append(action_index)
        rewards.append(result.reward)
    returns = np.zeros(len(rewards), dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + running
        returns[index] = running
    return np.asarray(observations), np.asarray(actions), returns, float(sum(rewards))


def _episode_log_probs(
    params: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
) -> jax.Array:
    """Log-probabilities for one episode, with hidden state reset at step zero."""
    hidden = jnp.zeros((1, params["w_xz"].shape[-1]), dtype=jnp.float32)

    def scan_step(hidden, inputs):
        observation, action = inputs
        raw, next_hidden = forward(params, observation[None, :], hidden, use_memory=True)
        return next_hidden, jax.nn.log_softmax(raw[0])[action]

    _, log_probs = jax.lax.scan(scan_step, hidden, (observations, actions))
    return log_probs


def _episode_log_probs_and_entropy(
    params: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Per-step log-probability and entropy for one episode."""
    hidden = jnp.zeros((1, params["w_xz"].shape[-1]), dtype=jnp.float32)

    def scan_step(hidden, inputs):
        observation, action = inputs
        raw, next_hidden = forward(params, observation[None, :], hidden, use_memory=True)
        log_probs = jax.nn.log_softmax(raw[0])
        probs = jax.nn.softmax(raw[0])
        entropy = -jnp.sum(probs * log_probs)
        return next_hidden, (log_probs[action], entropy)

    _, (log_probs, entropies) = jax.lax.scan(scan_step, hidden, (observations, actions))
    return log_probs, entropies


def _policy_loss(
    params: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
) -> jax.Array:
    """Policy-gradient loss over a batch of independent episodes.

    ``observations`` and ``actions`` have shape ``[episodes, steps, ...]``.
    Each episode starts from a fresh hidden state, so the loss does not depend
    on how episodes are ordered.
    """
    log_probs = jax.vmap(_episode_log_probs, in_axes=(None, 0, 0))(
        params, observations, actions
    )
    return -jnp.mean(log_probs * advantages)


def init_gravity_critic(seed: int, *, hidden: int = 16) -> dict[str, jax.Array]:
    """Small feed-forward state-value head over the encoded observation."""
    if hidden < 1:
        raise ValueError("hidden must be positive")
    keys = jax.random.split(jax.random.PRNGKey(seed + 99), 2)
    return {
        "w1": 0.1 * jax.random.normal(keys[0], (OBS_DIM, hidden)),
        "b1": jnp.zeros((hidden,)),
        "w2": 0.1 * jax.random.normal(keys[1], (hidden,)),
        "b2": jnp.zeros(()),
    }


def critic_values(
    critic: dict[str, jax.Array], observations: jax.Array
) -> jax.Array:
    """Value estimate for observations shaped ``[episodes, steps, obs_dim]``."""
    hidden = jnp.tanh(
        jnp.einsum("etd,dh->eth", observations, critic["w1"]) + critic["b1"]
    )
    return jnp.einsum("eth,h->et", hidden, critic["w2"]) + critic["b2"]


def fit_critic_step(
    critic: dict[str, jax.Array],
    observations: jax.Array,
    targets: jax.Array,
    *,
    learning_rate: float = 0.05,
) -> dict[str, jax.Array]:
    """One gradient step that fits the value head to observed returns."""

    def loss(head: dict[str, jax.Array]) -> jax.Array:
        return jnp.mean((critic_values(head, observations) - targets) ** 2)

    gradients = jax.grad(loss)(critic)
    return jax.tree_util.tree_map(
        lambda value, gradient: value - learning_rate * gradient, critic, gradients
    )


def _actor_critic_loss(
    actor: dict[str, jax.Array],
    critic: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    returns: jax.Array,
    entropy_coef: float,
    critic_coef: float,
) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
    """Advantage actor-critic loss with an entropy bonus.

    Returns ``(total, (actor_loss, critic_loss))``, where the second element is
    auxiliary output for diagnostics. The value head never leaks into the
    policy objective: advantages are stopped before the actor update.
    """
    values = critic_values(critic, observations)
    advantages = jax.lax.stop_gradient(returns - values)
    log_probs, entropies = jax.vmap(
        _episode_log_probs_and_entropy, in_axes=(None, 0, 0)
    )(actor, observations, actions)
    actor_loss = -jnp.mean(log_probs * advantages) - entropy_coef * jnp.mean(entropies)
    critic_loss = jnp.mean((values - returns) ** 2)
    return actor_loss + critic_coef * critic_loss, (actor_loss, critic_loss)


def train_gravity_controller(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1),
    training_treatments: tuple[Treatment, ...] = tuple(Treatment),
    updates: int = 4,
    episodes_per_update: int = 8,
    hidden: int = 8,
    learning_rate: float = 0.01,
    config: GravityConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    if not training_seeds or not training_treatments:
        raise ValueError("training seeds and treatments must not be empty")
    if updates < 1 or episodes_per_update < 1 or learning_rate <= 0:
        raise ValueError("invalid training configuration")
    config = config or GravityConfig()
    params = init_controller_params(
        1, OBS_DIM, len(_ACTIONS), hidden, jax.random.PRNGKey(seed)
    )
    specs = [
        (seed_value, treatment)
        for treatment in training_treatments
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 50_000)
    loss_and_grad = jax.jit(jax.value_and_grad(_policy_loss))
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(key, episodes_per_update + 1)
        key = keys[0]
        batches = [
            _collect_episode(params, *specs[index % len(specs)], config, keys[index + 1])
            for index in range(episodes_per_update)
        ]
        observations = np.stack([batch[0] for batch in batches])
        actions = np.stack([batch[1] for batch in batches])
        return_matrix = np.stack([batch[2] for batch in batches])
        advantages = return_matrix - return_matrix.mean(axis=0, keepdims=True)
        batch_advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)
        loss, gradients = loss_and_grad(
            params,
            jnp.asarray(observations),
            jnp.asarray(actions),
            jnp.asarray(batch_advantages),
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
        "status": "calibration_only",
        "trainer": "policy_gradient",
        "jax_loss_jitted": True,
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "training_seeds": list(training_seeds),
        "training_treatments": [treatment.value for treatment in training_treatments],
        "history": history,
    }


def train_gravity_actor_critic(
    seed: int,
    *,
    training_seeds: tuple[int, ...] = (0, 1),
    training_treatments: tuple[Treatment, ...] = tuple(Treatment),
    updates: int = 40,
    episodes_per_update: int = 16,
    hidden: int = 16,
    critic_hidden: int = 16,
    learning_rate: float = 0.02,
    entropy_coef: float = 0.05,
    critic_coef: float = 0.5,
    config: GravityConfig | None = None,
) -> tuple[dict[str, dict[str, jax.Array]], dict[str, Any]]:
    """Advantage actor-critic training with entropy control.

    Returns ``({"actor": ..., "critic": ...}, metadata)``. Advantage
    normalization happens per episode batch, and the critic predicts
    batch-normalized returns so its loss stays well scaled.
    """
    if not training_seeds or not training_treatments:
        raise ValueError("training seeds and treatments must not be empty")
    if updates < 1 or episodes_per_update < 2:
        raise ValueError("updates must be positive and episodes_per_update at least two")
    if learning_rate <= 0 or entropy_coef < 0 or critic_coef < 0:
        raise ValueError("invalid actor-critic hyperparameters")
    config = config or GravityConfig()
    actor = init_controller_params(
        1, OBS_DIM, len(_ACTIONS), hidden, jax.random.PRNGKey(seed)
    )
    critic = init_gravity_critic(seed, hidden=critic_hidden)
    specs = [
        (seed_value, treatment)
        for treatment in training_treatments
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 60_000)
    loss_and_grad = jax.jit(jax.value_and_grad(_actor_critic_loss, argnums=(0, 1), has_aux=True))
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(key, episodes_per_update + 1)
        key = keys[0]
        batches = [
            _collect_episode(actor, *specs[index % len(specs)], config, keys[index + 1])
            for index in range(episodes_per_update)
        ]
        observations = jnp.asarray(np.stack([batch[0] for batch in batches]))
        actions = jnp.asarray(np.stack([batch[1] for batch in batches]))
        returns = np.stack([batch[2] for batch in batches])
        # Predict batch-normalized returns, so the critic loss is well scaled.
        normalized = (returns - returns.mean()) / (returns.std() + 1e-6)
        normalized = jnp.asarray(normalized)
        (_, (actor_loss, critic_loss)), (actor_grads, critic_grads) = loss_and_grad(
            actor, critic, observations, actions, normalized, entropy_coef, critic_coef
        )
        actor = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * jnp.clip(gradient, -1.0, 1.0),
            actor,
            actor_grads,
        )
        critic = jax.tree_util.tree_map(
            lambda value, gradient: value - learning_rate * jnp.clip(gradient, -1.0, 1.0),
            critic,
            critic_grads,
        )
        history.append(
            {
                "update": update,
                "actor_loss": float(actor_loss),
                "critic_loss": float(critic_loss),
                "mean_return": float(np.mean([batch[3] for batch in batches])),
            }
        )
    return {"actor": actor, "critic": critic}, {
        "status": "calibration_only",
        "trainer": "actor_critic",
        "jax_loss_jitted": True,
        "seed": seed,
        "updates": updates,
        "episodes_per_update": episodes_per_update,
        "entropy_coef": entropy_coef,
        "critic_coef": critic_coef,
        "training_seeds": list(training_seeds),
        "training_treatments": [treatment.value for treatment in training_treatments],
        "history": history,
    }


def movement_rate(result: dict[str, Any]) -> float:
    """Fraction of episode steps that are not rest.

    In the gravity-present condition thrusting is useful.
    In the gravity-absent condition resting is free and thrusting is penalised.
    A constant policy therefore cannot show a difference between the two.
    """
    trace = result.get("agent_trace") or []
    if not trace:
        return 0.0
    return float(np.mean([item["action"] != Action.REST.value for item in trace]))


def detection_gap(
    present_results: list[dict[str, Any]],
    absent_results: list[dict[str, Any]],
) -> float:
    """Mean movement-rate difference between gravity present and absent.

    Zero means the organism behaves identically with and without gravity, so it
    has not detected whether the law is present. Positive values mean it rests
    more when gravity is absent, which is the adaptive response.
    """
    if not present_results or not absent_results:
        return 0.0
    present = float(np.mean([movement_rate(item) for item in present_results]))
    absent = float(np.mean([movement_rate(item) for item in absent_results]))
    return float(present - absent)


def measure_law_detection(
    params: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    horizon: int = 12,
) -> dict[str, Any]:
    """Compare behaviour with gravity present against gravity absent.

    Rewards are reported per condition for context, but they are not compared
    across conditions because the two conditions use different reward functions.
    The comparable quantity is the movement rate.
    """
    present_results: list[dict[str, Any]] = []
    absent_results: list[dict[str, Any]] = []
    per_direction: dict[str, dict[str, float]] = {}
    for direction in directions:
        config = GravityConfig(horizon=horizon, gravity_choices=(direction,))
        present_here: list[dict[str, Any]] = []
        absent_here: list[dict[str, Any]] = []
        for seed in seeds:
            present_here.append(
                run_gravity_episode(seed, Treatment.NO_REVELATION, params, config)
            )
            absent_here.append(
                run_gravity_episode(seed, Treatment.NO_BENEFIT, params, config)
            )
        present_results.extend(present_here)
        absent_results.extend(absent_here)
        per_direction[str(direction)] = {
            "present_movement_rate": float(
                np.mean([movement_rate(item) for item in present_here])
            ),
            "absent_movement_rate": float(
                np.mean([movement_rate(item) for item in absent_here])
            ),
            "present_reward": float(
                np.mean([item["total_reward"] for item in present_here])
            ),
            "absent_reward": float(
                np.mean([item["total_reward"] for item in absent_here])
            ),
        }
    return {
        "movement_gap": detection_gap(present_results, absent_results),
        "present_movement_rate": float(
            np.mean([movement_rate(item) for item in present_results])
        )
        if present_results
        else 0.0,
        "absent_movement_rate": float(
            np.mean([movement_rate(item) for item in absent_results])
        )
        if absent_results
        else 0.0,
        "per_direction": per_direction,
    }


def evaluate_gravity_controller(
    params: dict[str, jax.Array],
    *,
    seeds: tuple[int, ...] = (0, 1),
    config: GravityConfig | None = None,
) -> dict[str, Any]:
    config = config or GravityConfig()
    conditions: dict[str, dict[str, float]] = {}
    for treatment in Treatment:
        records = [run_gravity_episode(seed, treatment, params, config) for seed in seeds]
        conditions[treatment.value] = {
            "mean_reward": float(np.mean([record["total_reward"] for record in records])),
            "mean_gravity_alignment": float(
                np.mean([record["mean_gravity_alignment"] for record in records])
            ),
            "post_revelation_action_rate": float(
                np.mean([record["post_revelation_action_rate"] for record in records])
            ),
        }
    return {"status": "calibration_only", "conditions": conditions}


__all__ = [
    "init_linear_policy",
    "is_linear_policy",
    "train_linear_gravity_policy",
    "detection_gap",
    "encode_gravity_observation",
    "measure_law_detection",
    "movement_rate",
    "evaluate_gravity_controller",
    "run_gravity_episode",
    "train_gravity_controller",
]
