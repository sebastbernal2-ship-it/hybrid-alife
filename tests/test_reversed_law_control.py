"""The "reversed-law control" of the embodied law gate is not an independent control.

The held-out diagonals are closed under negation, so evaluating direction ``d`` with
``reverse_law=True`` is the same rollout as evaluating ``-d`` normally. See
``docs/results/embodied_law_null_selection/README.md``.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from hybrid_alife.experiments.latent_law_embodied import (
    alignment_gate_passes,
    embodied_law_config,
    per_direction_frozen_alignment,
)
from scripts.run_embodied_law_study import DIAGONALS

POPULATION = 16
CONFIG = embodied_law_config(
    0,
    population=POPULATION,
    size=14,
    proprioception=True,
    unit_flow_features=True,
    flow_head=True,
    displacement_reward=5.0,
)


def _genomes(gain: float, bias: tuple[float, float] = (0.0, 0.0)) -> dict:
    return {
        "w_flow": jnp.tile(gain * jnp.eye(2, dtype=jnp.float32), (POPULATION, 1, 1)),
        "b_flow": jnp.tile(jnp.asarray(bias, dtype=jnp.float32), (POPULATION, 1)),
    }


def _means(genomes: dict, reverse_law: bool = False) -> dict[str, float]:
    result = per_direction_frozen_alignment(
        None, CONFIG, DIAGONALS, genomes=genomes, reverse_law=reverse_law
    )
    return {key: value["mean_alignment"] for key, value in result.items()}


FIXED = _genomes(0.0, (3.0, 0.0))


def _antipodes(values: dict[str, float]) -> dict[str, float]:
    return {str(d): values[str((-d[0], -d[1]))] for d in DIAGONALS}


@pytest.fixture(scope="module")
def fixed_action() -> tuple[dict[str, float], dict[str, float]]:
    return _means(FIXED), _means(FIXED, reverse_law=True)


def test_reversed_law_is_the_antipodal_evaluation(fixed_action) -> None:
    evolved, reversed_law = fixed_action
    assert reversed_law == _antipodes(evolved)


@pytest.mark.xfail(
    strict=True,
    reason="documented defect: reversed-law values are a relabelling of the evolved values",
)
def test_reversed_law_is_an_independent_measurement(fixed_action) -> None:
    evolved, reversed_law = fixed_action
    assert sorted(reversed_law.values()) != sorted(evolved.values())


def test_fixed_action_needs_the_floor_the_reversed_clause_supplies(fixed_action) -> None:
    """The clause is not vacuous in delta mode: it is an evolved floor of 0.2.

    A fixed action beats an anti-aligned baseline by the margin on all four diagonals,
    including the two it moves against.
    """
    evolved, reversed_law = fixed_action
    initial = _means(_genomes(-3.0))
    assert all(evolved[key] - initial[key] >= 0.2 for key in evolved)
    assert min(evolved.values()) < -0.6
    assert not alignment_gate_passes(evolved, reversed_law, min_delta=0.2, initial=initial)
