from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_vivarium_sweep import aggregate, build_config, run_cell  # noqa: E402


def _args(**overrides):
    base = dict(population=8, world=12, season_length=4, steps=5, record_every=1)
    base.update(overrides)
    return argparse.Namespace(**base)


def test_build_config_carries_the_balance_parameters() -> None:
    config = build_config(0, 0.35, 0.03, _args())
    assert config.sim.embodied.predation_rate == 0.35
    assert config.sim.embodied.escape_gain == 0.03
    assert config.sim.embodied.max_energy == 80.0
    assert config.sim.embodied.body_morphology is True


def test_run_cell_smoke() -> None:
    row = run_cell(0, 0.35, 0.03, _args())
    assert row["predation_rate"] == 0.35
    assert row["escape_gain"] == 0.03
    assert row["final_population"] >= 0
    assert 0.0 <= row["death_rate"] <= 1.0


def test_aggregate_averages_over_seeds() -> None:
    rows = [
        {
            "predation_rate": 0.35,
            "escape_gain": 0.03,
            "tail_population": 100,
            "tail_energy": 10,
            "tail_body": 1.2,
            "death_rate": 0.01,
            "final_population": 100,
        },
        {
            "predation_rate": 0.35,
            "escape_gain": 0.03,
            "tail_population": 60,
            "tail_energy": 8,
            "tail_body": 1.0,
            "death_rate": 0.03,
            "final_population": 60,
        },
    ]
    out = aggregate(rows)
    assert len(out) == 1
    assert out[0]["tail_population"] == 80
    assert out[0]["seeds"] == 2
