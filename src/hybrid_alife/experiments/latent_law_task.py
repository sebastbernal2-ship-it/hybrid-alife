"""Calibration task for organisms exposed to a latent environmental law.

This module is separate from the field simulator.
It tests whether an organism can exploit gravity-like dynamics and declared
revelation treatments without receiving evaluator-only labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

import numpy as np


class Treatment(StrEnum):
    NO_REVELATION = "no_revelation"
    DEMONSTRATION = "demonstration"
    NOISY_SENSORY_HINT = "noisy_sensory_hint"
    SYMBOLIC_STATEMENT = "symbolic_statement"
    NO_BENEFIT = "no_benefit"


class Action(StrEnum):
    REST = "rest"
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    MOVE_UP = "move_up"
    MOVE_DOWN = "move_down"
    # Diagonal thrusts let the organism hold station against a diagonal gravity.
    # Without them an axis-only action set oscillates and cannot settle.
    MOVE_UP_LEFT = "move_up_left"
    MOVE_UP_RIGHT = "move_up_right"
    MOVE_DOWN_LEFT = "move_down_left"
    MOVE_DOWN_RIGHT = "move_down_right"


@dataclass(frozen=True)
class GravityConfig:
    horizon: int = 8
    gravity: tuple[float, float] = (0.0, -0.2)
    # When set, each episode draws one gravity vector from this declared set
    # instead of using ``gravity``. This makes a constant-action policy
    # insufficient, because the law changes between episodes.
    gravity_choices: tuple[tuple[float, float], ...] | None = None
    reversal_step: int | None = None
    revelation_step: int | None = None
    revelation_noise: float = 0.0
    revelation_cost: float = 0.0
    # Action acceleration must exceed the gravity magnitude so the organism can
    # brake and hold station instead of being pinned in a corner.
    action_acceleration: float = 0.25
    surface_friction: float = 0.9
    surface: float = 0.0
    world_limit: float = 6.0
    # Target radius along the gravity direction. It must lie well inside the
    # world so the target is reachable and stoppable.
    target_radius: float = 2.0
    # Reward design. "displacement" rewards movement along the gravity axis,
    # which is neutral at rest and improves incrementally. "target" rewards
    # distance to a target point and is retained for comparison only, because
    # its optimum is an isolated spike with no improvement path.
    reward_mode: str = "displacement"

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if len(self.gravity) != 2:
            raise ValueError("gravity must have two components")
        if self.gravity_choices is not None:
            if not self.gravity_choices:
                raise ValueError("gravity_choices must not be empty when set")
            if any(len(choice) != 2 for choice in self.gravity_choices):
                raise ValueError("every gravity choice must have two components")
        for name, step in (
            ("reversal_step", self.reversal_step),
            ("revelation_step", self.revelation_step),
        ):
            if step is not None and not 0 <= step < self.horizon:
                raise ValueError(f"{name} must be inside the episode")
        if not 0 <= self.revelation_noise < 1:
            raise ValueError("revelation_noise must be in [0, 1)")
        if self.revelation_cost < 0:
            raise ValueError("revelation_cost must be non-negative")
        if not 0 < self.surface_friction <= 1:
            raise ValueError("surface_friction must be in (0, 1]")
        if self.world_limit <= 0 or self.action_acceleration < 0:
            raise ValueError("world_limit must be positive and acceleration non-negative")
        if not 0 < self.target_radius < self.world_limit:
            raise ValueError("target_radius must lie inside the world")
        if self.reward_mode not in {"target", "displacement"}:
            raise ValueError("reward_mode must be 'target' or 'displacement'")


@dataclass(frozen=True)
class StepResult:
    reward: float
    agent_observation: dict[str, Any]
    evaluator_labels: dict[str, Any]
    revelation: dict[str, Any] | None


_DIAGONAL = float(1.0 / np.sqrt(2.0))
_ACTION_VECTORS = {
    Action.REST: np.asarray((0.0, 0.0), dtype=np.float64),
    Action.MOVE_LEFT: np.asarray((-1.0, 0.0), dtype=np.float64),
    Action.MOVE_RIGHT: np.asarray((1.0, 0.0), dtype=np.float64),
    Action.MOVE_UP: np.asarray((0.0, 1.0), dtype=np.float64),
    Action.MOVE_DOWN: np.asarray((0.0, -1.0), dtype=np.float64),
    Action.MOVE_UP_LEFT: np.asarray((-_DIAGONAL, _DIAGONAL), dtype=np.float64),
    Action.MOVE_UP_RIGHT: np.asarray((_DIAGONAL, _DIAGONAL), dtype=np.float64),
    Action.MOVE_DOWN_LEFT: np.asarray((-_DIAGONAL, -_DIAGONAL), dtype=np.float64),
    Action.MOVE_DOWN_RIGHT: np.asarray((_DIAGONAL, -_DIAGONAL), dtype=np.float64),
}


class GravityEpisode:
    """Deterministic two-dimensional organism episode with a latent gravity law."""

    def __init__(
        self,
        seed: int,
        treatment: Treatment | str,
        config: GravityConfig | None = None,
    ) -> None:
        self.seed = int(seed)
        self.treatment = Treatment(treatment)
        self.config = config or GravityConfig()
        self._rng = np.random.default_rng(self.seed)
        self._gravity_rng = np.random.default_rng(self.seed + 7)
        self.base_gravity = np.asarray(self._choose_gravity(), dtype=np.float64)
        self.step_index = 0
        self.position = np.zeros(2, dtype=np.float64)
        self.velocity = np.zeros(2, dtype=np.float64)
        self.revelation_log: list[dict[str, Any]] = []

    def _choose_gravity(self) -> tuple[float, float]:
        choices = self.config.gravity_choices
        if choices is None:
            return self.config.gravity
        index = int(self._gravity_rng.integers(0, len(choices)))
        return choices[index]

    @property
    def gravity_vector(self) -> tuple[float, float]:
        gravity = np.asarray(self.base_gravity, dtype=np.float64)
        if self.config.reversal_step is not None and self.step_index >= self.config.reversal_step:
            gravity = -gravity
        return tuple(float(value) for value in gravity)

    @property
    def _gravity_affects_outcome(self) -> bool:
        return self.treatment is not Treatment.NO_BENEFIT

    def _revelation_payload(self) -> dict[str, Any] | None:
        step = self.config.revelation_step
        if step is None or step != self.step_index:
            return None
        if self.treatment in (Treatment.NO_REVELATION, Treatment.NO_BENEFIT):
            return None
        gravity = np.asarray(self.gravity_vector, dtype=np.float64)
        if self.treatment is Treatment.DEMONSTRATION:
            return {
                "channel": "visual_demonstration",
                "content": {"velocity_change": tuple((gravity * 2).tolist())},
                "reliability": 1.0,
                "cost": self.config.revelation_cost,
            }
        if self.treatment is Treatment.NOISY_SENSORY_HINT:
            hint = gravity + self._rng.normal(0.0, self.config.revelation_noise, size=2)
            return {
                "channel": "noisy_sensor",
                "content": tuple(float(value) for value in hint),
                "reliability": 1.0 - self.config.revelation_noise,
                "cost": self.config.revelation_cost,
            }
        return {
            "channel": "external_message",
            "content": "external_statement: motion has a preferred acceleration direction",
            "reliability": 1.0,
            "cost": self.config.revelation_cost,
        }

    def _record_revelation(self, payload: dict[str, Any] | None) -> None:
        if payload is None or self.revelation_log:
            return
        self.revelation_log.append(
            {
                "step": self.step_index,
                "treatment": self.treatment.value,
                **payload,
            }
        )

    @property
    def agent_observation(self) -> dict[str, Any]:
        payload = self._revelation_payload()
        return {
            "step": self.step_index,
            "position": tuple(float(value) for value in self.position),
            "velocity": tuple(float(value) for value in self.velocity),
            "surface": float(self.config.surface),
            "revelation_present": payload is not None,
            "revelation": payload,
        }

    def _action_vector(self, action: Action | str) -> np.ndarray:
        action = Action(action)
        return _ACTION_VECTORS[action] * self.config.action_acceleration

    def step(self, action: Action | str) -> StepResult:
        action = Action(action)
        payload = self._revelation_payload()
        self._record_revelation(payload)
        action_vector = self._action_vector(action)
        gravity = np.asarray(self.gravity_vector, dtype=np.float64)
        applied_gravity = gravity if self._gravity_affects_outcome else np.zeros(2)
        self.velocity = (
            self.config.surface_friction * self.velocity + action_vector + applied_gravity
        )
        self.position = np.clip(
            self.position + self.velocity,
            -self.config.world_limit,
            self.config.world_limit,
        )
        if self._gravity_affects_outcome:
            direction = gravity / max(float(np.linalg.norm(gravity)), 1e-12)
            if self.config.reward_mode == "displacement":
                # Reward movement along the gravity axis. Rest is exactly zero
                # and every aligned step improves the total, so the optimum is
                # reachable by incremental improvement.
                speed = float(np.linalg.norm(action_vector))
                thrust = action_vector / speed if speed > 0 else action_vector
                reward = float(np.dot(thrust, direction))
            else:
                target = direction * self.config.target_radius
                reward = -float(np.linalg.norm(self.position - target))
        else:
            reward = -float(np.linalg.norm(action_vector))
        reward -= self.config.revelation_cost if payload is not None else 0.0
        self.step_index += 1
        next_gravity = self.gravity_vector
        labels = {
            "gravity_vector": next_gravity,
            "treatment": self.treatment.value,
            "surface": float(self.config.surface),
            "gravity_affects_outcome": self._gravity_affects_outcome,
            "step": self.step_index,
        }
        # ``_record_revelation`` is idempotent, so both the incoming payload and
        # the payload that becomes visible after the step are offered here.
        self._record_revelation(self._revelation_payload())
        return StepResult(
            reward=reward,
            agent_observation=self.agent_observation,
            evaluator_labels=labels,
            revelation=payload,
        )


def target_position(episode: GravityEpisode) -> np.ndarray:
    """Target location along the gravity direction. Evaluator-side utility."""
    gravity = np.asarray(episode.gravity_vector, dtype=np.float64)
    direction = gravity / max(float(np.linalg.norm(gravity)), 1e-12)
    return direction * episode.config.target_radius


def scripted_oracle_action(episode: GravityEpisode) -> Action:
    """Best action for a scripted baseline that knows gravity.

    Under the displacement reward this is simply the thrust most aligned with
    gravity. Under the target reward it is a greedy one-step lookahead.
    This is evaluator and test tooling and never the agent's controller.
    """
    gravity = np.asarray(episode.gravity_vector, dtype=np.float64)
    if episode.config.reward_mode == "displacement":
        direction = gravity / max(float(np.linalg.norm(gravity)), 1e-12)
        return max(
            _ACTION_VECTORS,
            key=lambda action: float(np.dot(_ACTION_VECTORS[action], direction)),
        )
    target = target_position(episode)
    friction = episode.config.surface_friction
    limit = episode.config.world_limit
    best_action = Action.REST
    best_distance = None
    for action, direction in _ACTION_VECTORS.items():
        velocity = (
            friction * episode.velocity
            + direction * episode.config.action_acceleration
            + gravity
        )
        position = np.clip(episode.position + velocity, -limit, limit)
        distance = float(np.linalg.norm(position - target))
        if best_distance is None or distance < best_distance:
            best_distance = distance
            best_action = action
    return best_action


__all__ = [
    "Action",
    "GravityConfig",
    "GravityEpisode",
    "StepResult",
    "Treatment",
    "scripted_oracle_action",
    "target_position",
]
