"""Tests for scripts/visualize_run.py.

Builds a synthetic run directory with metrics.jsonl + map_elites.npz +
novelty_archive.npz, runs the script via subprocess, and verifies that
the expected PNGs and index.html appear. Also exercises the
graceful-degradation path: a metrics-only run and an empty run dir
must both still produce a non-empty index.html.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "visualize_run.py"


def _write_metrics(path: Path, n: int = 12) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        rows.append(
            {
                "step": i,
                "generation": i,
                "embodied_alive_frac": 0.5 + 0.01 * i,
                "avida_alive_frac": 0.4 + 0.005 * i,
                "embodied_mean_lineage_depth": 1.0 + i * 0.1,
                "action_entropy": float(rng.uniform(0.2, 0.8)),
                "message_energy": float(rng.uniform(0.0, 1.0)),
                "comm_usage_rate": 0.2,
                "enrichment_separation": 0.1 + 0.02 * i,
                "coordinated_behavior_index": float(rng.uniform(0.0, 0.5)),
                "mean_avida_merit": 1.0 + i * 0.05,
                "qd_score": 10.0 + i,
                "map_elites_coverage": 0.05 + 0.01 * i,
                "novelty_archive_size": 10 + i,
            }
        )
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_map_elites(path: Path) -> None:
    rng = np.random.default_rng(1)
    fitness = rng.uniform(0.0, 1.0, size=(8, 8)).astype(np.float32)
    filled = (rng.uniform(size=(8, 8)) > 0.5).astype(bool)
    np.savez(path, fitness=fitness, filled=filled)


def _write_novelty_archive(path: Path) -> None:
    rng = np.random.default_rng(2)
    behaviors = rng.normal(size=(40, 2)).astype(np.float32)
    np.savez(path, behaviors=behaviors)


def _run(run_dir: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(run_dir), *args],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )


def test_full_run_produces_all_panels(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "metrics.jsonl")
    _write_map_elites(tmp_path / "map_elites.npz")
    _write_novelty_archive(tmp_path / "novelty_archive.npz")

    _run(tmp_path)

    viz = tmp_path / "viz"
    assert (viz / "index.html").exists()
    # at least the population/communication/lineage groups have data above
    timeseries = list(viz.glob("timeseries_*.png"))
    assert len(timeseries) >= 3, f"expected several timeseries PNGs, got {timeseries}"
    assert (viz / "map_elites.png").exists()
    assert (viz / "novelty_archive.png").exists()

    html = (viz / "index.html").read_text(encoding="utf-8")
    assert "MAP-Elites heatmap" in html
    assert "Novelty archive" in html
    assert "Time series" in html


def test_metrics_only_still_renders(tmp_path: Path) -> None:
    _write_metrics(tmp_path / "metrics.jsonl")
    _run(tmp_path, "--out-dir", str(tmp_path / "out"))
    out = tmp_path / "out"
    assert (out / "index.html").exists()
    assert any(out.glob("timeseries_*.png"))
    assert not (out / "map_elites.png").exists()
    assert not (out / "novelty_archive.png").exists()
    html = (out / "index.html").read_text(encoding="utf-8")
    assert "map_elites.npz not found" in html
    assert "novelty_archive.npz not found" in html


def test_empty_run_dir_still_writes_index(tmp_path: Path) -> None:
    _run(tmp_path)
    viz = tmp_path / "viz"
    html_path = viz / "index.html"
    assert html_path.exists()
    body = html_path.read_text(encoding="utf-8")
    assert "No visualizable artifacts" in body
