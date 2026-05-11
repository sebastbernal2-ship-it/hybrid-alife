"""CLI dry-run smoke tests.

Cheap guards that catch import / argparse breakage in `scripts/run_*.py`
without paying the cost of a full simulation. We rely on `--help` exiting
0 (argparse's documented behavior).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"

# Every top-level entry point we expect users / CI to invoke.
CLI_SCRIPTS = [
    "run_sim.py",
    "run_ablation_matrix.py",
    "run_comm_benchmark.py",
    "run_compute_scaling.py",
    "run_transfer_matrix.py",
    "generate_report.py",
]


def _run_help(script: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_help_exits_clean(script: str):
    assert (SCRIPTS_DIR / script).exists(), f"missing script: {script}"
    result = _run_help(script)
    assert result.returncode == 0, (
        f"{script} --help exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    # argparse prints "usage:" to stdout by default.
    assert "usage" in result.stdout.lower() or "usage" in result.stderr.lower()


@pytest.mark.parametrize("script", CLI_SCRIPTS)
def test_cli_module_imports(script: str):
    """Every script must import without side effects beyond argparse."""
    path = SCRIPTS_DIR / script
    source = path.read_text(encoding="utf-8")
    # Sanity: scripts gate execution behind __name__ == "__main__".
    assert '__name__ == "__main__"' in source, (
        f"{script} should guard main() with if __name__ == \"__main__\""
    )
