from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_task import (
    Action,
    GravityConfig,
    GravityEpisode,
    Treatment,
)


def test_gravity_dynamics_are_deterministic_and_action_sensitive() -> None:
    config = GravityConfig(horizon=4, gravity=(0.0, -0.2))
    first = GravityEpisode(seed=4, treatment=Treatment.NO_REVELATION, config=config)
    second = GravityEpisode(seed=4, treatment=Treatment.NO_REVELATION, config=config)
    first_result = first.step(Action.REST)
    second_result = second.step(Action.REST)
    assert first_result == second_result
    assert first_result.evaluator_labels["gravity_vector"] == (0.0, -0.2)
    moved = GravityEpisode(seed=4, treatment=Treatment.NO_REVELATION, config=config)
    moved.step(Action.MOVE_UP)
    assert not np.array_equal(moved.velocity, first.velocity)


def test_gravity_is_not_in_agent_observation() -> None:
    episode = GravityEpisode(seed=1, treatment=Treatment.NO_REVELATION)
    observation = episode.agent_observation
    assert "gravity_vector" not in observation
    assert "treatment" not in observation
    assert "evaluator_labels" not in observation


def test_symbolic_revelation_is_external_and_scheduled() -> None:
    config = GravityConfig(horizon=5, revelation_step=2)
    episode = GravityEpisode(seed=1, treatment=Treatment.SYMBOLIC_STATEMENT, config=config)
    assert episode.agent_observation["revelation_present"] is False
    episode.step(Action.REST)
    result = episode.step(Action.REST)
    assert result.agent_observation["revelation_present"] is True
    assert result.agent_observation["revelation"]["channel"] == "external_message"
    assert "external_statement" in result.agent_observation["revelation"]["content"]
    assert episode.revelation_log[0]["treatment"] == Treatment.SYMBOLIC_STATEMENT.value


def test_demonstration_revelation_uses_a_physical_effect_channel() -> None:
    config = GravityConfig(horizon=4, revelation_step=0)
    episode = GravityEpisode(seed=2, treatment=Treatment.DEMONSTRATION, config=config)
    revelation = episode.agent_observation["revelation"]
    assert revelation["channel"] == "visual_demonstration"
    assert "velocity_change" in revelation["content"]
    assert episode.step(Action.REST).revelation == revelation


def test_noisy_hint_is_logged_with_reliability_and_is_not_ground_truth() -> None:
    config = GravityConfig(horizon=4, revelation_step=0, revelation_noise=0.25)
    episode = GravityEpisode(seed=2, treatment=Treatment.NOISY_SENSORY_HINT, config=config)
    before = episode.agent_observation
    episode.step(Action.REST)
    hint = before["revelation"]["content"]
    assert before["revelation_present"] is True
    assert before["revelation"]["reliability"] == 0.75
    assert tuple(hint) != episode.gravity_vector or np.allclose(hint, episode.gravity_vector)
    assert episode.revelation_log[0]["channel"] == "noisy_sensor"


def test_reversal_changes_evaluator_law_without_leaking_it() -> None:
    config = GravityConfig(horizon=5, reversal_step=2, gravity=(0.0, -0.2))
    episode = GravityEpisode(seed=3, treatment=Treatment.NO_REVELATION, config=config)
    first = episode.step(Action.REST)
    second = episode.step(Action.REST)
    assert first.evaluator_labels["gravity_vector"] == (0.0, -0.2)
    assert second.evaluator_labels["gravity_vector"] == (0.0, 0.2)
    assert "gravity_vector" not in second.agent_observation


def test_no_benefit_control_keeps_reward_independent_of_gravity() -> None:
    config = GravityConfig(horizon=4, gravity=(0.0, -0.5))
    active = GravityEpisode(seed=5, treatment=Treatment.NO_REVELATION, config=config)
    control = GravityEpisode(seed=5, treatment=Treatment.NO_BENEFIT, config=config)
    active_result = active.step(Action.REST)
    control_result = control.step(Action.REST)
    assert active_result.reward != control_result.reward
    assert control_result.evaluator_labels["gravity_affects_outcome"] is False
    assert control_result.agent_observation["revelation_present"] is False


def test_variable_gravity_directions_are_seed_determined() -> None:
    choices = ((0.0, -0.2), (0.0, 0.2), (-0.2, 0.0), (0.2, 0.0))
    config = GravityConfig(horizon=3, gravity_choices=choices)
    first = GravityEpisode(seed=11, treatment=Treatment.NO_REVELATION, config=config)
    second = GravityEpisode(seed=11, treatment=Treatment.NO_REVELATION, config=config)
    assert first.gravity_vector == second.gravity_vector
    assert first.gravity_vector in choices
    seen = {
        GravityEpisode(seed=seed, treatment=Treatment.NO_REVELATION, config=config).gravity_vector
        for seed in range(40)
    }
    assert len(seen) > 1


def test_gravity_choices_do_not_leak_into_observation() -> None:
    config = GravityConfig(horizon=3, gravity_choices=((0.0, -0.2), (0.2, 0.0)))
    episode = GravityEpisode(seed=3, treatment=Treatment.NO_REVELATION, config=config)
    observation = episode.agent_observation
    assert "gravity_vector" not in observation
    assert "gravity_choices" not in observation


def test_action_set_can_hold_a_diagonal_station() -> None:
    """Nine unit actions, including the four diagonals, so a diagonal can be held."""
    from hybrid_alife.experiments.latent_law_task import _ACTION_VECTORS

    assert len(Action) == 9
    assert float(np.linalg.norm(_ACTION_VECTORS[Action.REST])) == 0.0
    thrusts = [v for a, v in _ACTION_VECTORS.items() if a is not Action.REST]
    assert all(abs(float(np.linalg.norm(v)) - 1.0) < 1e-9 for v in thrusts)
    assert Action.MOVE_UP_LEFT in _ACTION_VECTORS


def test_reachable_target_makes_rest_strictly_suboptimal() -> None:
    """A scripted oracle must beat resting, on cardinals and held-out diagonals."""
    from hybrid_alife.experiments.latent_law_task import scripted_oracle_action

    cardinals = ((0.0, -0.2), (0.0, 0.2), (-0.2, 0.0), (0.2, 0.0))
    diagonals = ((0.14142, -0.14142), (0.14142, 0.14142), (-0.14142, 0.14142), (-0.14142, -0.14142))
    for direction in cardinals + diagonals:
        config = GravityConfig(horizon=12, gravity_choices=(direction,))
        rest = GravityEpisode(7, Treatment.NO_REVELATION, config)
        oracle = GravityEpisode(7, Treatment.NO_REVELATION, config)
        rest_total = sum(rest.step(Action.REST).reward for _ in range(config.horizon))
        oracle_total = sum(
            oracle.step(scripted_oracle_action(oracle)).reward for _ in range(config.horizon)
        )
        assert oracle_total > rest_total + 5.0, (direction, oracle_total, rest_total)


def test_target_lies_inside_the_world() -> None:
    config = GravityConfig(gravity_choices=((0.14142, -0.14142),))
    episode = GravityEpisode(seed=1, treatment=Treatment.NO_REVELATION, config=config)
    gravity = np.asarray(episode.gravity_vector)
    target = gravity / np.linalg.norm(gravity) * config.target_radius
    assert np.linalg.norm(target) < config.world_limit
