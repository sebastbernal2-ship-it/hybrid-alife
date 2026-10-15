"""Explicit hidden-rule task wrapper for selective causal inquiry.

The task exposes only agent-visible observations through ``agent_observation``.
Ground-truth labels remain in ``StepResult.evaluator_labels``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np

SCHEMA_VERSION = 1


class WorldKind(StrEnum):
    SOLVABLE = "solvable"
    SURFACE_SHIFT = "surface_shift"
    REVERSAL = "reversal"
    DECEPTIVE = "deceptive"
    IMPOSSIBLE = "impossible"
    NON_IDENTIFIABLE = "non_identifiable"


class Action(StrEnum):
    EXPLOIT_A = "exploit_a"
    EXPLOIT_B = "exploit_b"
    PROBE = "probe"
    ABSTAIN = "abstain"
    TARGETED_A = "targeted_a"
    TARGETED_B = "targeted_b"


@dataclass(frozen=True)
class BenchmarkConfig:
    horizon: int = 8
    reversal_step: int = 4
    probe_cost: float = 0.1
    probe_accuracy: float = 1.0
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 0 < self.reversal_step < self.horizon:
            raise ValueError("reversal_step must be inside the episode")
        if self.probe_cost < 0:
            raise ValueError("probe_cost must be non-negative")
        if not 0.5 <= self.probe_accuracy <= 1.0:
            raise ValueError("probe_accuracy must be between 0.5 and 1.0")


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, int | None]
    reward: float
    correct: bool | None
    informative_probe: bool
    hidden_rule: int | None
    agent_observation: dict[str, int | None | bool]
    evaluator_labels: dict[str, int | None]


class CausalTaskEpisode:
    """A deterministic hidden-rule episode with explicit evaluator boundaries."""

    def __init__(self, kind: WorldKind | str, seed: int, config: BenchmarkConfig):
        self.kind = WorldKind(kind)
        self.seed = int(seed)
        self.config = config
        self._rng = np.random.default_rng(self.seed + 10_000)
        base_rng = np.random.default_rng(self.seed)
        self.base_rule = int(base_rng.integers(0, 2))
        self.surface = int(base_rng.integers(0, 4))
        self.probe_target = int(base_rng.integers(0, 2))
        self.cue = self._cue(base_rng)
        self.step_index = 0
        self.last_probe_result: int | None = None
        self.last_probe_informative = False

    def _cue(self, rng: np.random.Generator) -> int:
        if self.kind is WorldKind.DECEPTIVE:
            return self.base_rule
        if self.kind is WorldKind.NON_IDENTIFIABLE:
            return 0
        return int(rng.integers(0, 2))

    @property
    def current_rule(self) -> int | None:
        if self.kind is WorldKind.IMPOSSIBLE:
            return None
        if self.kind is WorldKind.REVERSAL and self.step_index >= self.config.reversal_step:
            return 1 - self.base_rule
        if self.kind is WorldKind.DECEPTIVE:
            return 1 - self.base_rule
        return self.base_rule

    @property
    def observation(self) -> dict[str, int | None]:
        return {
            "step": self.step_index,
            "surface": self.surface,
            "cue": self.cue,
            "probe_result": self.last_probe_result,
        }

    @property
    def agent_observation(self) -> dict[str, int | None | bool]:
        """Return the observation contract without evaluator-only labels."""
        observation = self.observation
        return {
            **observation,
            "cue_present": observation["cue"] is not None,
            "probe_result_present": observation["probe_result"] is not None,
        }

    def _probe(self, action: Action) -> tuple[int | None, bool]:
        rule = self.current_rule
        if self.kind is WorldKind.IMPOSSIBLE:
            return None, False
        if self.kind is WorldKind.NON_IDENTIFIABLE and action is Action.PROBE:
            return None, False
        if (
            self.kind is WorldKind.NON_IDENTIFIABLE
            and action in (Action.TARGETED_A, Action.TARGETED_B)
            and (action is Action.TARGETED_A) == bool(self.probe_target)
        ):
            return None, False
        if rule is None:
            return None, False
        if self.config.probe_accuracy < 1.0 and self._rng.random() >= self.config.probe_accuracy:
            return 1 - rule, True
        return rule, True

    def step(self, action: Action | str) -> StepResult:
        action = Action(action)
        hidden_rule = self.current_rule
        reward = 0.0
        correct: bool | None = None
        informative_probe = False
        self.last_probe_result = None
        self.last_probe_informative = False

        if action in (Action.PROBE, Action.TARGETED_A, Action.TARGETED_B):
            self.last_probe_result, informative_probe = self._probe(action)
            self.last_probe_informative = informative_probe
            reward = -self.config.probe_cost
        elif action is Action.ABSTAIN:
            pass
        else:
            selected = int(action is Action.EXPLOIT_B)
            if hidden_rule is None:
                outcome = int(self._rng.integers(0, 2))
                correct = selected == outcome
            else:
                correct = selected == hidden_rule
            reward = self.config.correct_reward if correct else self.config.incorrect_reward

        self.step_index += 1
        return StepResult(
            observation=self.observation,
            reward=float(reward),
            correct=correct,
            informative_probe=informative_probe,
            hidden_rule=hidden_rule,
            agent_observation=self.agent_observation,
            evaluator_labels={
                "hidden_rule": hidden_rule,
                "correct": int(correct) if correct is not None else None,
                "informative_probe": int(informative_probe),
            },
        )

    def oracle_action(self) -> Action:
        if self.current_rule is None:
            return Action.ABSTAIN
        if self.kind is WorldKind.NON_IDENTIFIABLE and self.last_probe_result is None:
            return Action.TARGETED_A if self.probe_target == 0 else Action.TARGETED_B
        return Action.EXPLOIT_B if self.current_rule else Action.EXPLOIT_A


__all__ = [
    "Action",
    "BenchmarkConfig",
    "CausalTaskEpisode",
    "SCHEMA_VERSION",
    "StepResult",
    "WorldKind",
]
