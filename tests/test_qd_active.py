"""Tests for the active MAP-Elites / QD path.

These exercise the descriptor-init -> archive-update -> per-gen logging
chain end-to-end on the tiny qd_active.yaml config, plus a focused unit
test on archive coverage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hybrid_alife.evolution.archives import MapElitesArchive
from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.metrics.qd import archive_entropy, coverage, qd_score


def test_map_elites_archive_update_covers_distinct_bins():
    arc = MapElitesArchive(bins=4)
    # Three descriptors that must land in three different bins.
    desc = np.array([[0.05, 0.05], [0.5, 0.5], [0.95, 0.95]], dtype=np.float32)
    fit = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    arc.update(desc, fit)
    assert arc.grid_filled.sum() == 3
    assert coverage(arc) > 0.0
    assert qd_score(arc) >= 0.0
    assert archive_entropy(arc) >= 0.0


def test_qd_active_run_produces_nonzero_coverage(tmp_path: Path):
    cfg_path = Path(__file__).resolve().parents[1] / "configs" / "qd_active.yaml"
    cfg = load_config(cfg_path)
    # Redirect output away from repo
    cfg.__dict__["output_dir"] = str(tmp_path)
    # Cap to a very small run for test speed; the config is already tiny
    # but we shrink further so this test stays well under a minute.
    cfg.evolution.__dict__["generations"] = 3
    cfg.evolution.__dict__["steps_per_generation"] = 4

    state = run_experiment(cfg)
    assert state.generation == cfg.evolution.generations - 1

    metrics_path = Path(cfg.output_dir) / cfg.run_name / "metrics.jsonl"
    assert metrics_path.exists()

    gen_records = []
    for line in metrics_path.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "generation":
            gen_records.append(rec)

    assert len(gen_records) == cfg.evolution.generations
    final = gen_records[-1]
    for key in ("map_elites_coverage", "qd_score", "archive_entropy"):
        assert key in final, f"missing {key} in per-generation record"
    # The whole point of qd_active.yaml: coverage must be strictly positive
    # after a short run.
    assert final["map_elites_coverage"] > 0.0
    assert final["archive_entropy"] >= 0.0
