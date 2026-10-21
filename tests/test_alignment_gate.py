from __future__ import annotations

from hybrid_alife.experiments.latent_law_embodied import alignment_gate_passes

DIRECTIONS = ("a", "b", "c", "d")


def _values(value: float) -> dict[str, float]:
    return {key: value for key in DIRECTIONS}


def test_absolute_floor_gate_passes_above_the_floor() -> None:
    assert alignment_gate_passes(
        _values(0.6), _values(0.5), min_evolved=0.5, reversed_min=0.2
    )


def test_absolute_floor_gate_fails_below_the_floor_on_any_direction() -> None:
    evolved = _values(0.6)
    evolved["c"] = 0.49
    assert not alignment_gate_passes(
        evolved, _values(0.5), min_evolved=0.5, reversed_min=0.2
    )


def test_absolute_floor_gate_fails_on_a_fixed_action_signature() -> None:
    reversed_law = _values(0.5)
    reversed_law["d"] = 0.1
    assert not alignment_gate_passes(
        _values(0.6), reversed_law, min_evolved=0.5, reversed_min=0.2
    )


def test_delta_gate_still_uses_the_initial_population() -> None:
    initial = _values(0.1)
    assert alignment_gate_passes(
        _values(0.6), _values(0.5), min_delta=0.2, initial=initial, reversed_min=0.2
    )
    initial["a"] = 0.5
    assert not alignment_gate_passes(
        _values(0.6), _values(0.5), min_delta=0.2, initial=initial, reversed_min=0.2
    )
