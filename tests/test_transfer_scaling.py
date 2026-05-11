"""Tests for the v1 transfer matrix and compute-scaling harness.

These exercise output-file creation and matrix/slope shape using synthetic
metrics so the suite stays CPU-cheap. Real-experiment integration is
verified separately via the smoke configs.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    """Import a top-level script module by path (scripts/ is not a package)."""
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Transfer matrix
# ---------------------------------------------------------------------------


def test_transfer_matrix_shape_and_files(tmp_path: Path) -> None:
    mod = _load_script("run_transfer_matrix")
    sources = {
        "src_a": {"action_entropy": 1.0, "mean_avida_merit": 2.0},
        "src_b": {"action_entropy": 1.5, "mean_avida_merit": 2.5},
    }
    targets = {
        "tgt_a": {"action_entropy": 1.2, "mean_avida_merit": 1.8},
        "tgt_b": {"action_entropy": 0.9, "mean_avida_merit": 2.1},
    }
    metrics = ["action_entropy", "mean_avida_merit"]
    matrix = mod.build_transfer_matrix(sources, targets, metrics)

    assert matrix["metrics"] == metrics
    assert matrix["sources"] == list(sources.keys())
    assert matrix["targets"] == list(targets.keys())
    assert len(matrix["cells"]) == len(sources) * len(targets)
    # Spot-check one delta: tgt_a.action_entropy - src_a.action_entropy = 0.2
    cell = next(c for c in matrix["cells"] if c["source"] == "src_a" and c["target"] == "tgt_a")
    assert cell["metrics"]["action_entropy"] == pytest.approx(0.2)
    assert cell["metrics"]["mean_avida_merit"] == pytest.approx(-0.2)

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    json_path = out_dir / "transfer_matrix.json"
    md_path = out_dir / "transfer_matrix.md"
    json_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    mod.write_matrix_markdown(matrix, md_path)

    assert json_path.exists()
    assert md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Transfer Matrix" in md
    assert "action_entropy" in md
    assert "tgt_a" in md and "src_a" in md


def test_transfer_matrix_handles_missing_metric() -> None:
    mod = _load_script("run_transfer_matrix")
    sources = {"s": {"a": 1.0}}
    targets = {"t": {"b": 2.0}}
    matrix = mod.build_transfer_matrix(sources, targets, ["a", "b"])
    cell = matrix["cells"][0]
    # No overlap → no delta entries
    assert cell["metrics"] == {}


# ---------------------------------------------------------------------------
# Compute scaling
# ---------------------------------------------------------------------------


def test_least_squares_slope_is_correct() -> None:
    mod = _load_script("run_compute_scaling")
    # y = 3x + 1 exactly → slope is 3
    xs = [0.0, 1.0, 2.0, 3.0]
    ys = [1.0, 4.0, 7.0, 10.0]
    assert mod.least_squares_slope(xs, ys) == pytest.approx(3.0)
    # Constant ys → slope is 0
    assert mod.least_squares_slope([1.0, 2.0], [5.0, 5.0]) == pytest.approx(0.0)
    # Degenerate xs → returns 0
    assert mod.least_squares_slope([1.0, 1.0], [1.0, 2.0]) == 0.0


def test_fit_slopes_shape_and_files(tmp_path: Path) -> None:
    mod = _load_script("run_compute_scaling")
    budgets = [10, 100, 1000]
    # log10 budgets = 1, 2, 3 → slope of 2*log10(b)+0.5 should be 2.0
    finals = [
        {"m": 2.5, "n": 0.0},
        {"m": 4.5, "n": 0.0},
        {"m": 6.5, "n": 0.0},
    ]
    summary = mod.fit_slopes(budgets, finals, ["m", "n", "missing"])
    assert summary["budgets"] == budgets
    assert set(summary["metrics"]) == {"m", "n", "missing"}
    assert summary["metrics"]["m"]["slope_per_log10_gen"] == pytest.approx(2.0)
    assert summary["metrics"]["n"]["slope_per_log10_gen"] == pytest.approx(0.0)
    # Missing metric: every row had no value → nan list → slope 0
    miss = summary["metrics"]["missing"]
    assert all(math.isnan(v) for v in miss["values"])
    assert miss["slope_per_log10_gen"] == 0.0

    out_dir = tmp_path / "out"
    out_dir.mkdir()
    json_path = out_dir / "scaling_slopes.json"
    md_path = out_dir / "scaling_slopes.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    mod.write_slopes_markdown(summary, md_path)
    assert json_path.exists() and md_path.exists()
    md = md_path.read_text(encoding="utf-8")
    assert "Compute Scaling Slopes" in md
    assert "slope/log10(gen)" in md


# ---------------------------------------------------------------------------
# Configs exist (sanity)
# ---------------------------------------------------------------------------


def test_placeholder_configs_exist() -> None:
    for name in [
        "configs/transfer_source.yaml",
        "configs/transfer_target_uniform.yaml",
        "configs/scaling_tiny.yaml",
    ]:
        assert (REPO_ROOT / name).is_file(), f"missing config: {name}"
