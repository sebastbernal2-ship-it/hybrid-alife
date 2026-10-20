"""Simulator-backed gravity-law bridge.

The law lives in the real shared world: the simulator's `flow` field.
A point organism feels that field as a force, and the protocol, observation,
reward, and learner are those of the calibration study:

- displacement reward: reward the thrust component along the local flow,
- proprioception: the organism sees its own last velocity and thrust,
- reward-weighted regression over a linear Gaussian thrust policy,
- trained on cardinal flow directions, evaluated on held-out diagonals.

This is not the evolved embodied GRU controller. It is the same question asked
inside the real world state, with a minimal organism, so the law comes from the
simulator rather than from a standalone environment.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.latent_law_continuous import (
    continuous_mean,
    fit_linear_thrust_policy,
    init_continuous_policy,
)
from hybrid_alife.types import SimState, WorldConfig
from hybrid_alife.world.env import initialize_world, step_world

FLOW_MAGNITUDE = 0.2


def world_config(*, width: int = 16, height: int = 16) -> WorldConfig:
    """A small, non-drifting world so an imposed flow stays fixed."""
    return WorldConfig(
        width=width,
        height=height,
        toroidal=True,
        resource_channels=1,
        hazard_channels=1,
        max_agents=8,
        max_digital_organisms=8,
        flow_noise_std=0.0,
        concentration_decay=0.99,
        concentration_diffusion=0.05,
        drifting=False,
    )


def initialize_law_state(
    seed: int,
    direction: tuple[float, float],
    *,
    config: WorldConfig | None = None,
    magnitude: float = FLOW_MAGNITUDE,
) -> SimState:
    """Build a real world and impose a uniform flow, the imposed law."""
    config = config or world_config()
    key = jax.random.PRNGKey(seed)
    world = initialize_world(config, key)
    unit = np.asarray(direction, dtype=np.float64)
    unit = unit / max(float(np.linalg.norm(unit)), 1e-12)
    flow = jnp.asarray(unit * magnitude, dtype=jnp.float32)
    world.flow = jnp.broadcast_to(flow, world.flow.shape)
    state = SimState(
        generation=0,
        step=0,
        rng=key,
        world=world,
        embodied=None,
        avida=None,
    )
    return state


class SimulatorOrganism:
    """A point organism moving under the real world's flow field."""

    def __init__(
        self,
        direction: tuple[float, float],
        *,
        horizon: int = 12,
        seed: int = 0,
        config: WorldConfig | None = None,
        magnitude: float = FLOW_MAGNITUDE,
        action_acceleration: float = 0.25,
        revelation: tuple[float, float] | None = None,
        revelation_step: int = 0,
        revelation_noise: float = 0.0,
        revelation_kind: str = "exact",
        reversal_step: int | None = None,
    ) -> None:
        if revelation_kind not in {"exact", "demonstration", "noisy", "symbolic"}:
            raise ValueError(
                "revelation_kind must be exact, demonstration, noisy, or symbolic"
            )
        self.revelation_kind = revelation_kind
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0 <= revelation_step < horizon:
            raise ValueError("revelation_step must be inside the episode")
        if revelation_noise < 0:
            raise ValueError("revelation_noise must be non-negative")
        if reversal_step is not None and not 0 <= reversal_step < horizon:
            raise ValueError("reversal_step must be inside the episode")
        self.reversal_step = reversal_step
        self.revelation = revelation
        self.revelation_step = revelation_step
        self.revelation_noise = revelation_noise
        self._revelation_seed = seed + 13
        self._revelation_cache: dict[int, tuple[float, float] | None] = {}
        if action_acceleration <= 0:
            raise ValueError("action_acceleration must be positive")
        self.horizon = horizon
        self.action_acceleration = action_acceleration
        self.magnitude = magnitude
        self.config = config or world_config()
        self.state = initialize_law_state(
            seed, direction, config=self.config, magnitude=magnitude
        )
        unit = np.asarray(direction, dtype=np.float64)
        self.law_direction = unit / max(float(np.linalg.norm(unit)), 1e-12)
        self.direction = direction
        self.step_index = 0
        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.previous_velocity = np.zeros(2, dtype=np.float64)
        self.previous_thrust = np.zeros(2, dtype=np.float64)

    def revelation_at(self, step_index: int) -> tuple[float, float] | None:
        """Revelation visible for the action taken at ``step_index``.

        Cached per step so repeated observation reads are deterministic.
        """
        if step_index in self._revelation_cache:
            return self._revelation_cache[step_index]
        if self.revelation is None or step_index != self.revelation_step:
            payload: tuple[float, float] | None = None
        else:
            value = np.asarray(self.revelation, dtype=np.float64)
            if self.revelation_kind == "symbolic":
                # The symbolic channel can only name a cardinal axis, so it is
                # lossy for diagonal laws. That is a real channel difference.
                axis = int(np.argmax(np.abs(value)))
                sign = float(np.sign(value[axis])) if value[axis] != 0 else 1.0
                value = np.zeros(2)
                value[axis] = sign
            if self.revelation_kind == "noisy" or self.revelation_noise > 0:
                rng = np.random.default_rng(self._revelation_seed + step_index)
                value = value + rng.normal(0.0, self.revelation_noise, size=2)
            payload = (float(value[0]), float(value[1]))
        self._revelation_cache[step_index] = payload
        return payload

    def local_flow(self) -> np.ndarray:
        """Flow at the organism's cell, read from the real world state."""
        height, width = self.state.world.flow.shape[0], self.state.world.flow.shape[1]
        y = int(np.clip((self.position[1] + 1.0) / 2.0 * (height - 1), 0, height - 1))
        x = int(np.clip((self.position[0] + 1.0) / 2.0 * (width - 1), 0, width - 1))
        return np.asarray(self.state.world.flow[y, x], dtype=np.float64)

    @property
    def agent_observation(self) -> dict[str, Any]:
        return {
            "step": self.step_index,
            "position": tuple(float(value) for value in self.position),
            "velocity": tuple(float(value) for value in self.velocity),
            "previous_velocity": tuple(float(value) for value in self.previous_velocity),
            "previous_thrust": tuple(float(value) for value in self.previous_thrust),
            "surface": 0.0,
            "revelation_present": self.revelation_at(self.step_index) is not None,
            "revelation": self.revelation_at(self.step_index),
        }

    def step(self, thrust: np.ndarray) -> float:
        """Advance the real world one step and move the organism."""
        # A reversal flips the imposed law in the real world state. The organism
        # is not told; it can only detect the change from its own sensors.
        if self.reversal_step is not None and self.step_index == self.reversal_step:
            self.state.world.flow = -self.state.world.flow
        self.state = step_world(self.state, self.config)
        thrust = np.asarray(thrust, dtype=np.float64).reshape(2)
        norm = float(np.linalg.norm(thrust))
        unit = thrust / norm if norm > 1e-9 else thrust
        flow = self.local_flow()
        flow_norm = float(np.linalg.norm(flow))
        flow_hat = flow / flow_norm if flow_norm > 1e-9 else flow
        reward = float(np.dot(unit, flow_hat))
        self.previous_velocity = np.array(self.velocity, dtype=np.float64)
        self.previous_thrust = np.array(unit, dtype=np.float64)
        self.velocity = (
            0.9 * self.velocity + unit * self.action_acceleration + flow
        )
        self.position = np.clip(self.position + self.velocity, -1.0, 1.0)
        self.step_index += 1
        return reward


def _collect_simulator_episode(
    policy: dict[str, jax.Array],
    direction: tuple[float, float],
    seed: int,
    key: jax.Array,
    *,
    horizon: int,
    exploration_std: float,
    config: WorldConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    organism = SimulatorOrganism(direction, horizon=horizon, seed=seed, config=config)
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    for _ in range(horizon):
        encoded = _encode(organism.agent_observation, horizon)
        mean = continuous_mean(policy, jnp.asarray(encoded))
        key, sample_key = jax.random.split(key)
        thrust = mean + exploration_std * jax.random.normal(sample_key, mean.shape)
        clipped = np.asarray(_unit(np.asarray(thrust)))
        observations.append(encoded)
        actions.append(clipped)
        rewards.append(organism.step(clipped))
    return np.asarray(observations), np.asarray(actions), float(sum(rewards))


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-9 else vector


def _encode(observation: dict[str, Any], horizon: int) -> np.ndarray:
    from hybrid_alife.experiments.latent_law_controller import (
        encode_gravity_observation,
    )

    return encode_gravity_observation(observation, horizon)


def run_simulator_episode(
    params: dict[str, jax.Array],
    direction: tuple[float, float],
    *,
    seed: int = 0,
    horizon: int = 12,
    config: WorldConfig | None = None,
) -> dict[str, Any]:
    """Deterministic deployment of the mean thrust inside the real world."""
    organism = SimulatorOrganism(direction, horizon=horizon, seed=seed, config=config)
    rewards: list[float] = []
    for _ in range(horizon):
        encoded = _encode(organism.agent_observation, horizon)
        thrust = np.asarray(continuous_mean(params, jnp.asarray(encoded)))
        rewards.append(organism.step(_unit(thrust)))
    return {
        "direction": direction,
        "total_reward": float(sum(rewards)),
        "rewards": rewards,
    }


def train_simulator_policy_rwr(
    seed: int,
    *,
    training_directions: tuple[tuple[float, float], ...],
    training_seeds: tuple[int, ...] = (0, 1, 2, 3),
    rounds: int = 24,
    episodes_per_round: int = 64,
    horizon: int = 12,
    exploration_std: float = 0.6,
    exploration_std_final: float = 0.05,
    temperature: float = 2.0,
    ridge: float = 1e-3,
    config: WorldConfig | None = None,
) -> tuple[dict[str, jax.Array], dict[str, Any]]:
    """Reward-weighted regression against the simulator-imposed flow law."""
    if not training_directions or not training_seeds:
        raise ValueError("training directions and seeds must not be empty")
    if rounds < 1 or episodes_per_round < 1:
        raise ValueError("rounds and episodes_per_round must be positive")
    policy = init_continuous_policy(seed)
    specs = [
        (direction, seed_value)
        for direction in training_directions
        for seed_value in training_seeds
    ]
    key = jax.random.PRNGKey(seed + 88_000)
    history: list[dict[str, float]] = []
    for round_index in range(rounds):
        progress = round_index / max(rounds - 1, 1)
        current_std = exploration_std + progress * (
            exploration_std_final - exploration_std
        )
        keys = jax.random.split(key, episodes_per_round + 1)
        key = keys[0]
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        totals: list[float] = []
        for index in range(episodes_per_round):
            direction, seed_value = specs[index % len(specs)]
            obs, act, total = _collect_simulator_episode(
                policy,
                direction,
                seed_value,
                keys[index + 1],
                horizon=horizon,
                exploration_std=current_std,
                config=config,
            )
            observations.append(obs)
            actions.append(act)
            totals.append(total)
        stacked_observations = np.concatenate(observations, axis=0)
        stacked_actions = np.concatenate(actions, axis=0)
        weights = np.exp((np.asarray(totals) - np.max(totals)) / temperature)
        step_weights = np.repeat(weights, horizon)
        policy = fit_linear_thrust_policy(
            stacked_observations, stacked_actions, step_weights, ridge
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
        "trainer": "simulator_reward_weighted_regression",
        "seed": seed,
        "rounds": rounds,
        "episodes_per_round": episodes_per_round,
        "horizon": horizon,
        "exploration_std": exploration_std,
        "exploration_std_final": exploration_std_final,
        "temperature": temperature,
        "training_directions": [list(direction) for direction in training_directions],
        "history": history,
    }


def per_direction_simulator_rewards(
    params: dict[str, jax.Array],
    directions: tuple[tuple[float, float], ...],
    seeds: tuple[int, ...],
    *,
    horizon: int = 12,
    config: WorldConfig | None = None,
) -> dict[str, float]:
    """Deterministic simulator reward, reported per imposed-flow direction."""
    scores: dict[str, float] = {}
    for direction in directions:
        values = [
            run_simulator_episode(
                params, direction, seed=seed, horizon=horizon, config=config
            )["total_reward"]
            for seed in seeds
        ]
        scores[str(direction)] = float(np.mean(values))
    return scores


__all__ = [
    "FLOW_MAGNITUDE",
    "SimulatorOrganism",
    "initialize_law_state",
    "per_direction_simulator_rewards",
    "run_simulator_episode",
    "train_simulator_policy_rwr",
    "world_config",
]
