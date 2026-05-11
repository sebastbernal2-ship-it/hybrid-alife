#!/usr/bin/env python
"""Run an ablation matrix and produce a transfer/robustness summary.

Drives the existing single-run `run_experiment` over a list of ablation configs
times a list of seeds, then writes a markdown summary using
`hybrid_alife.experiments.transfer.write_summary_markdown`.

Designed to be CPU-feasible on a smoke budget: by default each cell runs only
a handful of generations from `configs/smoke200.yaml`-style configs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import List

import yaml

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.experiments.shadow import run_shadow
from hybrid_alife.experiments.transfer import AblationResult, write_summary_markdown
from hybrid_alife.logging.jsonl import read_jsonl
from hybrid_alife.types import ExperimentConfig


DEFAULT_METRICS = [
    "embodied_alive_frac",
    "embodied_lineage_hill1d",
    "action_entropy",
    "comm_usage_rate",
    "mean_avida_merit",
    "avida_tasks_solved",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/smoke200.yaml",
            "configs/ablation_no_comms.yaml",
            "configs/ablation_static_world.yaml",
        ],
    )
    p.add_argument("--seeds", type=int, default=3, help="Seeds per config.")
    p.add_argument(
        "--include-shadow",
        action="store_true",
        help="Run a paired neutral-shadow (no-selection) version of the first config.",
    )
    p.add_argument("--out-dir", type=str, default="outputs/ablation_matrix")
    p.add_argument(
        "--baseline-name",
        type=str,
        default=None,
        help="Name (run_name from the config) to use as the baseline for "
        "Cliff's δ effect sizes. Defaults to the first config's run_name.",
    )
    return p.parse_args()


def _load_with_seed(cfg_path: str, seed: int, out_dir: str) -> ExperimentConfig:
    cfg = load_config(cfg_path)
    cfg = replace(cfg, seed=seed, run_name=f"{cfg.run_name}_s{seed}", output_dir=out_dir)
    return cfg


def _last_metric_row(run_dir: Path) -> dict:
    metrics = run_dir / "metrics.jsonl"
    if not metrics.exists():
        return {}
    records = read_jsonl(metrics)
    return records[-1] if records else {}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: List[AblationResult] = []
    baseline_name: str | None = args.baseline_name

    for cfg_path in args.configs:
        base_cfg = load_config(cfg_path)
        name = base_cfg.run_name
        if baseline_name is None:
            baseline_name = name
        for seed in range(args.seeds):
            cfg = _load_with_seed(cfg_path, seed, str(out_dir))
            run_experiment(cfg)
            row = _last_metric_row(out_dir / cfg.run_name)
            results.append(AblationResult(name=name, seed=seed, metrics=row))

    if args.include_shadow:
        shadow_base = load_config(args.configs[0])
        for seed in range(args.seeds):
            cfg = _load_with_seed(args.configs[0], seed, str(out_dir))
            run_shadow(cfg)
            row = _last_metric_row(out_dir / f"{cfg.run_name}_shadow")
            results.append(AblationResult(name=f"{shadow_base.run_name}_shadow", seed=seed, metrics=row))

    summary_path = out_dir / "ablation_summary.md"
    write_summary_markdown(results, summary_path, DEFAULT_METRICS, baseline=baseline_name)

    # Also dump the raw results as JSON for downstream processing.
    json_path = out_dir / "ablation_results.json"
    json_path.write_text(
        json.dumps([{"name": r.name, "seed": r.seed, "metrics": r.metrics} for r in results], indent=2),
        encoding="utf-8",
    )
    print(f"Ablation summary written to {summary_path}")
    print(f"Raw results written to {json_path}")


if __name__ == "__main__":
    main()
