"""Calibration environments for selective causal inquiry.

This module is a deterministic validity layer, not a consciousness test.
It checks whether an environment family can distinguish causal discovery,
active probing, reversal adaptation, transfer, and calibrated abstention before
expensive hybrid-alife campaigns run.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

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
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0

    def __post_init__(self) -> None:
        if self.horizon < 2:
            raise ValueError("horizon must be at least 2")
        if not 0 < self.reversal_step < self.horizon:
            raise ValueError("reversal_step must be inside the episode")
        if self.probe_cost < 0:
            raise ValueError("probe_cost must be non-negative")


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, int | None]
    reward: float
    correct: bool | None
    informative_probe: bool
    hidden_rule: int | None


Policy = Callable[[dict[str, int | None], dict[str, Any]], Action]


class HiddenRuleEpisode:
    """A small causal environment with hidden rules and explicit probes."""

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

    def _probe(self, action: Action) -> tuple[int | None, bool]:
        rule = self.current_rule
        if self.kind is WorldKind.IMPOSSIBLE:
            return None, False
        if self.kind is WorldKind.NON_IDENTIFIABLE and action is Action.PROBE:
            return None, False
        if (
            self.kind is WorldKind.NON_IDENTIFIABLE
            and action in (Action.TARGETED_A, Action.TARGETED_B)
            and (action is Action.TARGETED_A) != bool(self.probe_target)
        ):
            return None, False
        return rule, rule is not None

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
                # The impossible world has seeded outcomes independent of action.
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
        )

    def oracle_action(self) -> Action:
        if self.current_rule is None:
            return Action.ABSTAIN
        if self.kind is WorldKind.NON_IDENTIFIABLE and self.last_probe_result is None:
            return Action.TARGETED_A if self.probe_target == 0 else Action.TARGETED_B
        return Action.EXPLOIT_B if self.current_rule else Action.EXPLOIT_A


def _exploit(rule: int | None) -> Action:
    if rule is None:
        return Action.ABSTAIN
    return Action.EXPLOIT_B if rule else Action.EXPLOIT_A


def _cue_policy(observation: dict[str, int | None], _memory: dict[str, Any]) -> Action:
    result = observation["probe_result"]
    if result is not None:
        return _exploit(result)
    cue = observation["cue"]
    return _exploit(cue) if cue is not None else Action.ABSTAIN


def _probe_then_exploit_policy(
    observation: dict[str, int | None], memory: dict[str, Any]
) -> Action:
    result = observation["probe_result"]
    if result is not None:
        memory["rule"] = result
        return _exploit(result)
    if observation["step"] == 0:
        return Action.PROBE
    return _exploit(memory.get("rule"))


def _targeted_probe_policy(
    observation: dict[str, int | None], memory: dict[str, Any]
) -> Action:
    result = observation["probe_result"]
    if result is not None:
        memory["rule"] = result
        return _exploit(result)
    if observation["step"] == 0:
        return Action.TARGETED_A
    if observation["step"] == 1 and "target_tried" not in memory:
        memory["target_tried"] = True
        return Action.TARGETED_B
    return _exploit(memory.get("rule"))


def _reversal_probe_policy(
    observation: dict[str, int | None], memory: dict[str, Any]
) -> Action:
    result = observation["probe_result"]
    if result is not None:
        memory["rule"] = result
        return _exploit(result)
    if observation["step"] in (0, 4):
        return Action.PROBE
    return _exploit(memory.get("rule"))


def _random_policy(observation: dict[str, int | None], memory: dict[str, Any]) -> Action:
    rng = memory.setdefault("rng", np.random.default_rng(memory.get("seed", 0) + 50_000))
    return Action(rng.choice(list(Action)))


def _abstain_policy(_observation: dict[str, int | None], _memory: dict[str, Any]) -> Action:
    return Action.ABSTAIN


_POLICIES: dict[str, Policy] = {
    "cue": _cue_policy,
    "probe_then_exploit": _probe_then_exploit_policy,
    "targeted_probe": _targeted_probe_policy,
    "reversal_probe": _reversal_probe_policy,
    "random": _random_policy,
    "abstain": _abstain_policy,
}


def _run_episode(
    kind: WorldKind,
    seed: int,
    policy: str,
    config: BenchmarkConfig,
) -> dict[str, Any]:
    episode = HiddenRuleEpisode(kind, seed, config)
    memory: dict[str, Any] = {"seed": seed}
    rewards: list[float] = []
    exploit_count = 0
    correct_exploits = 0
    probe_count = 0
    informative_probe_count = 0
    first_post_reversal_correct: int | None = None
    initial_rule = episode.current_rule
    for _ in range(config.horizon):
        if policy == "oracle":
            action = episode.oracle_action()
        else:
            action = _POLICIES[policy](episode.observation, memory)
        result = episode.step(action)
        rewards.append(result.reward)
        if action in (Action.EXPLOIT_A, Action.EXPLOIT_B):
            exploit_count += 1
            correct_exploits += int(result.correct is True)
        if action in (Action.PROBE, Action.TARGETED_A, Action.TARGETED_B):
            probe_count += 1
            informative_probe_count += int(result.informative_probe)
        if (
            kind is WorldKind.REVERSAL
            and episode.step_index > config.reversal_step
            and result.correct is True
            and first_post_reversal_correct is None
        ):
            first_post_reversal_correct = episode.step_index - config.reversal_step
    final_rule = episode.current_rule
    return {
        "world": kind.value,
        "seed": seed,
        "policy": policy,
        "total_reward": float(sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "probe_count": probe_count,
        "probe_rate": float(probe_count / config.horizon),
        "informative_probe_count": informative_probe_count,
        "information_gain": float(informative_probe_count),
        "correct_exploit_rate": float(correct_exploits / exploit_count) if exploit_count else 0.0,
        "structure_rate": float(exploit_count / config.horizon),
        "abstain_rate": float(
            sum(1 for reward in rewards if reward == 0.0) / config.horizon
        ),
        "reversal_recovery_steps": first_post_reversal_correct,
        "rule_before_reversal": initial_rule,
        "rule_after_reversal": final_rule,
    }


def run_episode(
    kind: WorldKind | str,
    seed: int,
    policy: str,
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Run one deterministic episode with a named baseline policy."""
    config = config or BenchmarkConfig()
    if policy != "oracle" and policy not in _POLICIES:
        raise ValueError(f"unknown policy: {policy}")
    return _run_episode(WorldKind(kind), seed, policy, config)


def _mean(records: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([record[key] for record in records])) if records else 0.0


def validate_environment_ladder(
    seeds: tuple[int, ...] = (0, 1, 2),
    config: BenchmarkConfig | None = None,
) -> dict[str, Any]:
    """Check that the ladder separates learnable, impossible, and probe worlds."""
    config = config or BenchmarkConfig()
    solvable_oracle = [run_episode(WorldKind.SOLVABLE, seed, "oracle", config) for seed in seeds]
    solvable_no_signal = [
        run_episode(WorldKind.SOLVABLE, seed, "abstain", config) for seed in seeds
    ]
    impossible_probe = [
        run_episode(WorldKind.IMPOSSIBLE, seed, "probe_then_exploit", config) for seed in seeds
    ]
    surface_rules = [
        HiddenRuleEpisode(WorldKind.SOLVABLE, seed, config).base_rule
        == HiddenRuleEpisode(WorldKind.SURFACE_SHIFT, seed, config).base_rule
        for seed in seeds
    ]
    reversal_rules = [
        (
            HiddenRuleEpisode(WorldKind.REVERSAL, seed, config).base_rule
            != (
                1 - HiddenRuleEpisode(WorldKind.REVERSAL, seed, config).base_rule
            )
        )
        for seed in seeds
    ]
    generic = [
        run_episode(WorldKind.NON_IDENTIFIABLE, seed, "probe_then_exploit", config)
        for seed in seeds
    ]
    targeted = [
        run_episode(WorldKind.NON_IDENTIFIABLE, seed, "targeted_probe", config)
        for seed in seeds
    ]
    checks = {
        "oracle_beats_no_signal": _mean(solvable_oracle, "total_reward")
        > _mean(solvable_no_signal, "total_reward"),
        "impossible_has_no_signal": all(
            record["information_gain"] == 0.0 for record in impossible_probe
        ),
        "surface_shift_preserves_rule": all(surface_rules),
        "reversal_changes_rule": all(reversal_rules),
        "non_identifiable_requires_targeted_probe": _mean(targeted, "information_gain")
        > _mean(generic, "information_gain"),
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "details": {
            "solvable_oracle_mean_reward": _mean(solvable_oracle, "total_reward"),
            "solvable_no_signal_mean_reward": _mean(solvable_no_signal, "total_reward"),
            "impossible_probe_mean_information_gain": _mean(
                impossible_probe, "information_gain"
            ),
            "non_identifiable_generic_information_gain": _mean(
                generic, "information_gain"
            ),
            "non_identifiable_targeted_information_gain": _mean(
                targeted, "information_gain"
            ),
        },
    }


def evaluate_benchmark(
    seeds: tuple[int, ...] = (0, 1, 2),
    horizon: int = 8,
) -> dict[str, Any]:
    """Run all calibration worlds and baselines with fixed seeds."""
    config = BenchmarkConfig(horizon=horizon)
    policies = ("oracle", "cue", "probe_then_exploit", "targeted_probe", "random", "abstain")
    worlds = {
        kind.value: [
            run_episode(kind, seed, policy, config)
            for seed in seeds
            for policy in policies
        ]
        for kind in WorldKind
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "seeds": list(seeds),
        "config": {
            "horizon": config.horizon,
            "reversal_step": config.reversal_step,
            "probe_cost": config.probe_cost,
        },
        "validity_gate": validate_environment_ladder(seeds, config),
        "worlds": worlds,
        "status": "calibration_only",
    }
