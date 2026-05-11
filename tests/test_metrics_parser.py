"""Metrics-parser and JSONL round-trip sanity checks.

These verify the surface used by `scripts/generate_report.py` and other
downstream consumers: writing a metrics stream as JSONL and reading it back
yields exactly what we wrote, including numeric types we care about.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hybrid_alife.logging.jsonl import JsonlWriter, _json_default, read_jsonl


def test_jsonl_writer_handles_numpy_scalars(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    with JsonlWriter(p) as w:
        w.write({"step": np.int32(3), "loss": np.float32(0.25)})
        w.write({"step": np.int64(4), "loss": np.float64(0.125)})
    rows = read_jsonl(p)
    assert rows == [
        {"step": 3, "loss": 0.25},
        {"step": 4, "loss": 0.125},
    ]


def test_jsonl_writer_handles_numpy_arrays(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    with JsonlWriter(p) as w:
        w.write({"vec": np.array([1.0, 2.0, 3.0])})
    rows = read_jsonl(p)
    assert rows == [{"vec": [1.0, 2.0, 3.0]}]


def test_json_default_falls_back_to_string(tmp_path: Path):
    """Unknown objects must serialize as strings rather than crashing the
    metrics writer mid-run — losing some fidelity is better than dropping
    the entire metrics line."""

    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    assert _json_default(Weird()) == "<weird>"

    # End-to-end: an unknown object inside a row must not raise.
    p = tmp_path / "m.jsonl"
    with JsonlWriter(p) as w:
        w.write({"obj": Weird(), "ok": 1})
    rows = read_jsonl(p)
    assert rows[0]["ok"] == 1
    assert isinstance(rows[0]["obj"], str)


def test_jsonl_reader_skips_blank_lines(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    p.write_text(
        '{"a": 1}\n\n{"a": 2}\n   \n{"a": 3}\n',
        encoding="utf-8",
    )
    rows = read_jsonl(p)
    assert rows == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_jsonl_round_trip_preserves_order(tmp_path: Path):
    p = tmp_path / "m.jsonl"
    payload = [{"i": i, "v": float(i) ** 0.5} for i in range(50)]
    with JsonlWriter(p) as w:
        for row in payload:
            w.write(row)
    rows = read_jsonl(p)
    assert rows == payload


def test_jsonl_lines_are_valid_json(tmp_path: Path):
    """Every emitted line must independently round-trip through json.loads.
    Guards against accidental multiline records."""
    p = tmp_path / "m.jsonl"
    with JsonlWriter(p) as w:
        w.write({"a": 1, "b": "hello"})
        w.write({"a": 2, "b": "world"})
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        assert "a" in obj and "b" in obj


def test_collect_full_metrics_returns_finite_floats():
    """`collect_full_metrics` is the surface `run_experiment` writes each
    generation. Every value should be a finite Python float — anything else
    will silently corrupt downstream report generation."""
    from hybrid_alife.experiments.runner import initialize_sim, load_config
    from hybrid_alife.metrics.core import collect_full_metrics

    cfg = load_config("configs/base.yaml")
    state = initialize_sim(cfg)
    metrics = collect_full_metrics(state)
    assert metrics, "expected a non-empty metrics dict"
    for k, v in metrics.items():
        # Allow ints too (e.g. counts) but reject NaN / Inf.
        assert isinstance(v, (int, float)), f"{k}: non-numeric value {v!r}"
        assert np.isfinite(float(v)), f"{k}: non-finite value {v!r}"
