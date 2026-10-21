from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_d1_ablation import CONDITIONS, FULL_FLAGS, build_command, summarize  # noqa: E402


def test_every_ablation_changes_one_mechanism_from_full() -> None:
    full = set(FULL_FLAGS)
    assert set(CONDITIONS) == {
        "full",
        "no_flow_head",
        "raw_flow",
        "energy_fitness",
        "no_rank",
        "no_immigrants",
    }
    assert set(CONDITIONS["full"]) == full
    # Each ablation differs from full by dropping or replacing a named mechanism.
    assert "--flow-head" not in CONDITIONS["no_flow_head"]
    assert "--linear-policy" in CONDITIONS["no_flow_head"]
    assert "--unit-flow-features" not in CONDITIONS["raw_flow"]
    assert "--raw-flow-features" in CONDITIONS["raw_flow"]
    assert "min_alignment" not in CONDITIONS["energy_fitness"]
    assert "--rank-replacement" not in CONDITIONS["no_rank"]
    assert "--rank-replacement" in CONDITIONS["no_immigrants"]


def test_build_command_carries_the_budget_and_condition() -> None:
    command = build_command(
        "full", seeds=[0, 1], generations=3, output=Path("/tmp/out.json")
    )
    assert "--generations" in command
    assert command[command.index("--generations") + 1] == "3"
    assert command[command.index("--seeds") + 1 : command.index("--generations")] == [
        "0",
        "1",
    ]
    assert "--flow-head" in command


def test_summarize_reports_worst_case_values() -> None:
    summary = {
        "seeds_passed": 1,
        "seeds_total": 2,
        "results": [
            {"seed": 0, "passed": True, "min_delta": 0.3, "min_evolved": 0.6, "min_reversed": 0.5},
            {"seed": 1, "passed": False, "min_delta": 0.1, "min_evolved": 0.4, "min_reversed": 0.2},
        ],
    }
    row = summarize(summary)
    assert row["seeds_passed"] == 1
    assert row["worst_delta"] == 0.1
    assert row["worst_evolved"] == 0.4
    assert row["worst_reversed"] == 0.2
