from __future__ import annotations

import jax.numpy as jnp

from hybrid_alife.experiments.latent_law_inference import run_inference_episode
from hybrid_alife.experiments.latent_law_simulator import SimulatorOrganism

# Thrust = estimate + revelation when present.
REVEAL = {
    "w": jnp.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    "b": jnp.zeros((2,)),
}
# Thrust = revelation only; the estimate is ignored.
MESSAGE_ONLY = {
    "w": jnp.asarray([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
    "b": jnp.zeros((2,)),
}
HORIZON = 12


def test_symbolic_channel_quantises_to_a_cardinal_axis() -> None:
    organism = SimulatorOrganism(
        (0.7071, -0.7071),
        horizon=4,
        seed=0,
        revelation=(0.7071, -0.7071),
        revelation_kind="symbolic",
    )
    payload = organism.revelation_at(0)
    assert payload is not None
    assert abs(payload[0]) + abs(payload[1]) == 1.0
    # An exact diagonal is a tie on both axes; the first axis wins.
    assert payload == (1.0, 0.0)
    skewed = SimulatorOrganism(
        (0.3, -0.9),
        horizon=4,
        seed=0,
        revelation=(0.3, -0.9),
        revelation_kind="symbolic",
    )
    assert skewed.revelation_at(0) == (0.0, -1.0)


def test_demonstration_channel_matches_the_exact_law() -> None:
    organism = SimulatorOrganism(
        (0.0, -1.0),
        horizon=4,
        seed=0,
        revelation=(0.0, -1.0),
        revelation_kind="demonstration",
    )
    assert organism.revelation_at(0) == (0.0, -1.0)


def test_noisy_channel_is_perturbed_and_repeatable() -> None:
    organism = SimulatorOrganism(
        (0.0, -1.0),
        horizon=4,
        seed=0,
        revelation=(0.0, -1.0),
        revelation_kind="noisy",
        revelation_noise=0.1,
    )
    first = organism.revelation_at(0)
    assert first is not None
    assert first != (0.0, -1.0)
    assert first == organism.revelation_at(0)


def test_true_revelation_forms_reach_the_full_optimum() -> None:
    direction = (0.0, -1.0)
    for kind in ("exact", "demonstration"):
        result = run_inference_episode(
            REVEAL,
            direction,
            horizon=HORIZON,
            use_revelation=True,
            revelation=direction,
        )
        assert result["total_reward"] > 11.9, (kind, result["total_reward"])


def test_symbolic_channel_is_lossy_on_a_diagonal_law() -> None:
    direction = (0.7071, -0.7071)
    exact = run_inference_episode(
        REVEAL, direction, horizon=HORIZON, use_revelation=True, revelation=direction
    )
    symbolic = run_inference_episode(
        REVEAL,
        direction,
        horizon=HORIZON,
        use_revelation=True,
        revelation=direction,
        revelation_step=0,
    )
    assert exact["total_reward"] > symbolic["total_reward"] - 0.01


def test_message_only_control_is_insufficient() -> None:
    direction = (0.0, -1.0)
    result = run_inference_episode(
        MESSAGE_ONLY,
        direction,
        horizon=HORIZON,
        use_revelation=True,
        revelation=direction,
    )
    # One point from one message step, then nothing without the estimate.
    assert result["total_reward"] < 2.0


def test_false_revelation_self_corrects_within_one_step() -> None:
    direction = (1.0, 0.0)
    result = run_inference_episode(
        REVEAL,
        direction,
        horizon=HORIZON,
        use_revelation=True,
        revelation=(-1.0, 0.0),
    )
    assert 8.9 < result["total_reward"] < 11.1
