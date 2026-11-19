from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_sweep_rescored as an  # noqa: E402

DIRS = ("d0", "d1")


def _rows(trained_by_drag: dict, init_by_drag: dict, seed: int = 0) -> dict:
    """Build rows[(kind, drag, genomes, seed)] with a single target value per direction."""
    rows: dict = {}
    for drag, vals in trained_by_drag.items():
        rows[("own", drag, "trained", seed)] = {
            d: {t: v for t in an.TARGETS} for d, v in zip(DIRS, vals, strict=True)
        }
    for drag, vals in init_by_drag.items():
        rows[("own", drag, "init", seed)] = {
            d: {t: v for t in an.TARGETS} for d, v in zip(DIRS, vals, strict=True)
        }
    return rows


def test_holm_matches_hand_computation_and_is_monotone():
    adj = an.holm([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])
    assert (an.holm([0.5]) == [0.5]).all()


def test_per_seed_slope_recovers_a_known_line():
    drags = np.array(an.DRAGS)
    assert abs(an.per_seed_slope(drags, 0.4 - 0.07 * drags) + 0.07) < 1e-12


def test_min_delta_takes_the_worst_direction_in_the_same_fluid():
    rows = _rows({0.5: (0.6, 0.5)}, {0.5: (0.1, 0.4)})
    # directions: 0.6-0.1 = 0.5 and 0.5-0.4 = 0.1 -> min is 0.1
    assert abs(an.min_delta(rows, "own", 0.5, "official", 0) - 0.1) < 1e-12


def test_transfer_uses_the_uncoupled_initial_reference():
    rows = _rows({0.5: (0.9, 0.9)}, {0.0: (0.2, 0.3), 0.5: (0.8, 0.8)})
    rows[("transfer", 0.5, "trained", 0)] = {
        "d0": {t: 0.7 for t in an.TARGETS},
        "d1": {t: 0.6 for t in an.TARGETS},
    }
    # transfer minus the drag-0 initial: 0.7-0.2 = 0.5 and 0.6-0.3 = 0.3
    assert abs(an.min_delta(rows, "transfer", 0.5, "official", 0) - 0.3) < 1e-12


def test_load_rows_groups_by_seed_kind_drag_genomes(tmp_path):
    row = {"kind": "own", "drag": 0.5, "genomes": "init", "direction": "d0", "official": 0.1}
    (tmp_path / "seed3.json").write_text(json.dumps({"seed": 3, "rows": [row]}))
    rows = an.load_rows(tmp_path)
    assert rows[("own", 0.5, "init", 3)]["d0"]["official"] == 0.1


def test_bootstrap_interval_contains_the_mean_of_a_constant_sample():
    lo, hi = an.boot_ci(np.full(15, 0.3), np.random.default_rng(0))
    assert abs(lo - 0.3) < 1e-12 and abs(hi - 0.3) < 1e-12
