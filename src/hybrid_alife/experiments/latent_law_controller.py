"""Calibration controller for the latent gravity-law task."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.agents.controller import forward, init_controller_params
from hybrid_alife.experiments.latent_law_task import (
    Action,
    GravityConfig,
    GravityEpisode,
    Treatment,
)

OBS_DIM = 13
_ACTIONS = tuple(Action)
_ACTION_VECTORS = {
    Action.REST: np.asarray((0.0, 0.0)),
    Action.MOVE_LEFT: np.asarray((-1.0, 0.0)),
    Action.MOVE_RIGHT: np.asarray((1.0, 0.0)),
    Action.MOVE_UP: np.asarray((0.0, 1.0)),
    Action.MOVE_DOWN: np.asarray((0.0, -1.0)),
}


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
    hidden = jnp.zeros((1, int(params["w_xz"].shape[-1])), dtype=jnp.float32)
    agent_trace: list[dict[str, Any]] = []
    evaluator_trace: list[dict[str, Any]] = []
    rewards: list[float] = []
    for _ in range(config.horizon):
        observation = dict(episode.agent_observation)
        encoded = encode_gravity_observation(observation, config.horizon)
        raw, hidden = forward(params, jnp.asarray(encoded)[None, :], hidden, use_memory=True)
        action = _action_from_logits(np.asarray(raw[0]))
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


def _loss(
    params: dict[str, jax.Array],
    observations: jax.Array,
    actions: jax.Array,
    advantages: jax.Array,
) -> jax.Array:
    hidden = jnp.zeros((1, params["w_xz"].shape[-1]), dtype=jnp.float32)

    def scan_step(hidden, inputs):
        observation, action = inputs
        raw, next_hidden = forward(params, observation[None, :], hidden, use_memory=True)
        log_probs = jax.nn.log_softmax(raw[0])
        return next_hidden, log_probs[action]

    _, log_probs = jax.lax.scan(scan_step, hidden, (observations, actions))
    return -jnp.mean(log_probs * advantages)


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
    loss_and_grad = jax.jit(jax.value_and_grad(_loss))
    history: list[dict[str, float | int]] = []
    for update in range(updates):
        keys = jax.random.split(key, episodes_per_update + 1)
        key = keys[0]
        batches = [
            _collect_episode(params, *specs[index % len(specs)], config, keys[index + 1])
            for index in range(episodes_per_update)
        ]
        observations = np.concatenate([batch[0] for batch in batches])
        actions = np.concatenate([batch[1] for batch in batches])
        return_matrix = np.stack([batch[2] for batch in batches])
        advantages = return_matrix - return_matrix.mean(axis=0, keepdims=True)
        flat_advantages = advantages.reshape(-1)
        flat_advantages = (flat_advantages - flat_advantages.mean()) / (
            flat_advantages.std() + 1e-6
        )
        loss, gradients = loss_and_grad(
            params,
            jnp.asarray(observations),
            jnp.asarray(actions),
            jnp.asarray(flat_advantages),
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
    "encode_gravity_observation",
    "evaluate_gravity_controller",
    "run_gravity_episode",
    "train_gravity_controller",
]
