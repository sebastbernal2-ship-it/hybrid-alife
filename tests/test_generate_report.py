"""Tests for scripts/generate_report.py — make sure partial run dirs render
without crashing, and that key sections appear when their artifacts exist.

These tests invoke the script via subprocess so they exercise the same CLI
surface users see.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_report.py"


def _write_metrics(path: Path, n: int = 5) -> None:
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "step": i,
                    "generation": i,
                    "embodied_alive_frac": 0.5 + 0.01 * i,
                    "comm_usage_rate": 0.2,
                }
            )
            for i in range(n)
        )
        + "\n",
        encoding="utf-8",
    )


def _run(run_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )


def test_minimal_run_dir_renders(tmp_path: Path) -> None:
    """A run dir with only metrics.jsonl should still produce a report."""
    _write_metrics(tmp_path / "metrics.jsonl")
    _run(tmp_path, "--no-plots")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# Experiment Report:" in report
    assert "Headline metrics" in report
    assert "Scientific caveats" in report
    assert "No `scaling_slopes.json` present" in report
    assert "No `transfer_matrix.json` present" in report


def test_full_artifact_set(tmp_path: Path) -> None:
    """All optional artifacts present → all sections populate."""
    _write_metrics(tmp_path / "metrics.jsonl")
    (tmp_path / "scaling_slopes.json").write_text(
        json.dumps(
            {"axes": {"population": {"slope": 0.5, "r2": 0.9, "n": 5}}}
        )
    )
    (tmp_path / "transfer_matrix.json").write_text(
        json.dumps({"tasks": ["A", "B"], "matrix": [[1.0, 0.3], [0.4, 1.0]]})
    )
    _run(tmp_path, "--no-plots")
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Compute-scaling slopes" in report
    assert "population" in report
    assert "POET-style transfer matrix" in report
    assert "| A |" in report


def test_headline_only(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "metrics.jsonl")
    _run(tmp_path, "--headline", "--out", "headline.md")
    report = (tmp_path / "headline.md").read_text(encoding="utf-8")
    assert "Headline metrics:" in report
    # Headline mode should NOT include the caveats narrative block
    assert "Scientific caveats" not in report


def test_baseline_diff(tmp_path: Path) -> None:
    run = tmp_path / "run"
    baseline = tmp_path / "baseline"
    run.mkdir()
    baseline.mkdir()
    _write_metrics(run / "metrics.jsonl", n=10)
    _write_metrics(baseline / "metrics.jsonl", n=10)
    _run(run, "--headline", "--baseline", str(baseline))
    report = (run / "report.md").read_text(encoding="utf-8")
    assert "baseline avg" in report
    # Identical sequences → Δ should be 0.0000 for embodied_alive_frac
    assert "0.0000" in report
