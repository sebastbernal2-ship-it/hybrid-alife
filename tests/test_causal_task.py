from __future__ import annotations

from hybrid_alife.experiments.causal_task import (
    Action,
    BenchmarkConfig,
    CausalTaskEpisode,
    WorldKind,
)


def test_agent_observation_has_presence_bits_but_no_hidden_rule() -> None:
    episode = CausalTaskEpisode(WorldKind.SOLVABLE, seed=3, config=BenchmarkConfig())
    observation = episode.agent_observation
    assert set(observation) == {
        "step",
        "surface",
        "cue",
        "cue_present",
        "probe_result",
        "probe_result_present",
    }
    assert "hidden_rule" not in observation
    step = episode.step(Action.PROBE)
    assert "hidden_rule" not in step.agent_observation
    assert step.evaluator_labels["hidden_rule"] in (0, 1)


def test_noisy_probe_is_declared_and_deterministic() -> None:
    config = BenchmarkConfig(horizon=6, probe_accuracy=0.75)
    first = CausalTaskEpisode(WorldKind.SOLVABLE, seed=3, config=config)
    second = CausalTaskEpisode(WorldKind.SOLVABLE, seed=3, config=config)
    assert first.step(Action.PROBE).agent_observation == second.step(Action.PROBE).agent_observation
    assert first.last_probe_result in (0, 1)


def test_non_identifiable_target_is_visible_only_as_probe_result() -> None:
    episode = CausalTaskEpisode(
        WorldKind.NON_IDENTIFIABLE,
        seed=3,
        config=BenchmarkConfig(),
    )
    generic = episode.step(Action.PROBE)
    assert generic.agent_observation["probe_result_present"] is False
    target = Action.TARGETED_A if episode.probe_target == 0 else Action.TARGETED_B
    targeted = episode.step(target)
    assert targeted.agent_observation["probe_result_present"] is True
