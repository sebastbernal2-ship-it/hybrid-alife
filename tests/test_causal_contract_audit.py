from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from hybrid_alife.experiments.awareness_benchmark import (
    encode_observation,
    init_learned_controller,
)
from hybrid_alife.experiments.causal_controller import _policy_gradient_loss
from hybrid_alife.experiments.causal_task import (
    Action,
    BenchmarkConfig,
    CausalTaskEpisode,
    WorldKind,
)


def test_impossible_world_makes_abstention_the_expected_non_probe_choice() -> None:
    config = BenchmarkConfig(horizon=4, reversal_step=2)
    exploit_rewards = []
    abstain_rewards = []
    for seed in range(100):
        exploit = CausalTaskEpisode(WorldKind.IMPOSSIBLE, seed, config)
        abstain = CausalTaskEpisode(WorldKind.IMPOSSIBLE, seed, config)
        exploit_rewards.append(exploit.step(Action.EXPLOIT_A).reward)
        abstain_rewards.append(abstain.step(Action.ABSTAIN).reward)
    assert abs(np.mean(exploit_rewards)) < 0.3
    assert np.mean(abstain_rewards) == 0.0
    probe = CausalTaskEpisode(WorldKind.IMPOSSIBLE, 0, config)
    assert probe.step(Action.PROBE).reward == -config.probe_cost


def test_deceptive_cue_disagrees_with_hidden_rule() -> None:
    episode = CausalTaskEpisode(WorldKind.DECEPTIVE, 4, BenchmarkConfig())
    assert episode.agent_observation["cue"] != episode.current_rule
    assert "hidden_rule" not in episode.agent_observation


def test_probe_cost_and_result_boundary_are_explicit() -> None:
    episode = CausalTaskEpisode(WorldKind.SOLVABLE, 4, BenchmarkConfig())
    before = episode.agent_observation
    assert before["probe_result_present"] is False
    result = episode.step(Action.PROBE)
    assert result.reward == -episode.config.probe_cost
    assert result.agent_observation["probe_result_present"] is True
    assert "hidden_rule" not in result.agent_observation
    assert result.evaluator_labels["hidden_rule"] in (0, 1)


def test_one_step_policy_gradient_moves_toward_positive_return_action() -> None:
    params = init_learned_controller(3, hidden=4)
    observation = encode_observation(
        CausalTaskEpisode(WorldKind.SOLVABLE, 0, BenchmarkConfig()).agent_observation
    )
    observations = jnp.asarray(observation[None, :])
    actions = jnp.asarray([tuple(Action).index(Action.EXPLOIT_A)])
    returns = jnp.asarray([1.0])
    loss, gradients = jax.value_and_grad(_policy_gradient_loss)(
        params, observations, actions, returns, 0.0
    )
    updated = jax.tree_util.tree_map(
        lambda value, gradient: value - 0.1 * gradient,
        params,
        gradients,
    )
    updated_loss = _policy_gradient_loss(
        updated, observations, actions, returns, 0.0
    )
    assert np.isfinite(float(loss))
    assert float(updated_loss) < float(loss)
