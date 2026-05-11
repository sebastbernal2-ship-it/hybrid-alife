#!/usr/bin/env python
"""V1 transfer matrix for hybrid-alife.

For each (source_config, target_config) pair, we run both configs (optionally
re-using existing run dirs via --source-runs / --target-runs) and report the
final-row metric deltas (target - source). This is *checkpoint-level
transfer*: it measures how the headline metric shifts when the world changes,
not full policy generalisation. See docs/poet_transfer.md for limitations.

Outputs (under --out-dir):

    transfer_matrix.json
    transfer_matrix.md

The JSON shape is documented in docs/poet_transfer.md so downstream tooling
and tests can rely on it.

Typical usage::

    python scripts/run_transfer_matrix.py \
        --source-configs configs/transfer_source.yaml \
        --target-configs configs/transfer_source.yaml \
                         configs/transfer_target_uniform.yaml \
        --out-dir outputs/transfer_matrix
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.logging.jsonl import read_jsonl
from hybrid_alife.types import ExperimentConfig


DEFAULT_METRICS = [
    "action_entropy",
    "mean_avida_merit",
    "embodied_alive_frac",
    "comm_usage_rate",
    "avida_tasks_solved",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--source-configs", nargs="+", required=True)
    p.add_argument("--target-configs", nargs="+", required=True)
    p.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_METRICS,
        help="Metric names to include in the matrix.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/transfer_matrix")
    p.add_argument(
        "--skip-run-if-exists",
        action="store_true",
        help="If a run dir with metrics.jsonl already exists, reuse it.",
    )
    return p.parse_args()


def _run_and_collect(
    cfg_path: str, seed: int, out_dir: Path, skip_if_exists: bool
) -> tuple[str, dict[str, float]]:
    """Run a single config under ``out_dir`` and return (run_name, final_row)."""
    cfg = load_config(cfg_path)
    run_name = f"{cfg.run_name}_s{seed}"
    cfg = replace(cfg, seed=seed, run_name=run_name, output_dir=str(out_dir))
    run_dir = out_dir / run_name
    metrics_path = run_dir / "metrics.jsonl"
    if not (skip_if_exists and metrics_path.exists()):
        run_experiment(cfg)
    rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    final = rows[-1] if rows else {}
    return cfg.run_name, _clean_floats(final)


def _clean_floats(record: dict) -> dict[str, float]:
    out: dict[str, float] = {}
    for k, v in record.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def build_transfer_matrix(
    source_metrics: dict[str, dict[str, float]],
    target_metrics: dict[str, dict[str, float]],
    metrics: Sequence[str],
) -> dict:
    """Build the matrix JSON object from per-config final-metric dicts."""
    cells = []
    for source, src_row in source_metrics.items():
        for target, tgt_row in target_metrics.items():
            delta: dict[str, float] = {}
            for m in metrics:
                s = src_row.get(m)
                t = tgt_row.get(m)
                if s is None or t is None:
                    continue
                delta[m] = float(t) - float(s)
            cells.append({"source": source, "target": target, "metrics": delta})
    return {
        "metrics": list(metrics),
        "sources": list(source_metrics.keys()),
        "targets": list(target_metrics.keys()),
        "cells": cells,
        "source_finals": source_metrics,
        "target_finals": target_metrics,
    }


def write_matrix_markdown(matrix: dict, out_path: Path) -> None:
    lines = ["# Transfer Matrix (target − source deltas)\n"]
    metrics = matrix["metrics"]
    for m in metrics:
        lines.append(f"## {m}\n")
        header = "| source \\\\ target | " + " | ".join(matrix["targets"]) + " |"
        sep = "|---|" + "|".join(["---:"] * len(matrix["targets"])) + "|"
        lines.append(header)
        lines.append(sep)
        by_pair = {(c["source"], c["target"]): c["metrics"] for c in matrix["cells"]}
        for source in matrix["sources"]:
            cells = []
            for target in matrix["targets"]:
                v = by_pair.get((source, target), {}).get(m)
                cells.append("n/a" if v is None else f"{v:+.4f}")
            lines.append(f"| {source} | " + " | ".join(cells) + " |")
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    source_metrics: dict[str, dict[str, float]] = {}
    for cfg_path in args.source_configs:
        name, row = _run_and_collect(cfg_path, args.seed, out_dir, args.skip_run_if_exists)
        source_metrics[name] = row

    target_metrics: dict[str, dict[str, float]] = {}
    for cfg_path in args.target_configs:
        name, row = _run_and_collect(cfg_path, args.seed, out_dir, args.skip_run_if_exists)
        target_metrics[name] = row

    matrix = build_transfer_matrix(source_metrics, target_metrics, args.metrics)

    json_path = out_dir / "transfer_matrix.json"
    md_path = out_dir / "transfer_matrix.md"
    json_path.write_text(json.dumps(matrix, indent=2), encoding="utf-8")
    write_matrix_markdown(matrix, md_path)
    print(f"Transfer matrix written to {json_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
