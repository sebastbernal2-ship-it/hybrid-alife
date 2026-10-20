from __future__ import annotations

import numpy as np

from hybrid_alife.experiments.latent_law_embodied import (
    HIDDEN_GROUPS,
    embodied_law_config,
    per_direction_frozen_alignment,
    run_embodied_law_pilot,
    signed_alignment,
)

CARDINALS = ((0.0, -1.0), (0.0, 1.0), (1.0, 0.0), (-1.0, 0.0))
DIAGONALS = ((0.7071, -0.7071), (0.7071, 0.7071))


def test_hidden_groups_hide_the_flow_from_observation() -> None:
    config = embodied_law_config(seed=0)
    assert "flow" in config.embodied.hidden_field_groups
    assert set(HIDDEN_GROUPS).issubset(set(config.embodied.hidden_field_groups))
    # The law still couples physically and is still rewarded.
    assert config.embodied.flow_movement_scale > 0
    assert config.embodied.flow_alignment_reward > 0
    assert config.avida.enabled is False


def test_signed_alignment_excludes_empty_and_zero_motion_steps() -> None:
    moves = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]])
    flows = np.asarray([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
    alive = np.asarray([True, True, True, True])
    # Aligned +1, anti -1; the rest step and the no-flow step are excluded.
    assert abs(signed_alignment(moves, flows, alive) - 0.0) < 1e-9
    assert abs(signed_alignment(moves[:2], flows[:2], alive[:2])) < 1e-9
    assert signed_alignment(moves[3:], flows[3:], alive[3:]) == 0.0
    dead = np.asarray([False, True])
    assert signed_alignment(moves[:2], flows[:2], dead) == -1.0


def test_bounded_pilot_runs_and_reports_per_direction() -> None:
    pilot = run_embodied_law_pilot(
        seed=0,
        directions=CARDINALS,
        generations=2,
        population=8,
        steps_per_generation=2,
        size=10,
    )
    assert pilot["status"] == "pilot_only"
    assert len(pilot["training_alignment"]) == 2
    scores = per_direction_frozen_alignment(
        pilot["state"], pilot["config"], DIAGONALS, steps=4
    )
    assert set(scores) == {str(d) for d in DIAGONALS}
    for value in scores.values():
        assert np.isfinite(value["mean_alignment"])
        assert -1.0 <= value["mean_alignment"] <= 1.0
