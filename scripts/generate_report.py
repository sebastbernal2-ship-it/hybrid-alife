#!/usr/bin/env python
"""Generate a markdown report from an experiment run directory.

Usage:
    python scripts/generate_report.py outputs/runs/smoke200
"""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean

from hybrid_alife.logging.jsonl import read_jsonl
from hybrid_alife.viz.plots import (
    plot_agent_positions,
    plot_map_elites,
    plot_metrics,
    plot_world_fields,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    final_ckpt = run_dir / "checkpoint_final.pkl"
    map_elites = run_dir / "map_elites.npz"

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    field_plot = plot_world_fields(final_ckpt, plots_dir / "world_fields.png")
    agents_plot = plot_agent_positions(final_ckpt, plots_dir / "agents.png")
    metrics_plot = plot_metrics(metrics_path, plots_dir / "metrics.png")
    me_plot = plot_map_elites(map_elites, plots_dir / "map_elites.png")

    records = read_jsonl(metrics_path)
    n = len(records)
    last = records[-1] if records else {}

    def avg(key: str) -> float:
        vals = [r[key] for r in records if key in r]
        return mean(vals) if vals else float("nan")

    lines = [
        f"# Experiment Report: {run_dir.name}",
        "",
        f"- Records logged: {n}",
        f"- Generations reached: {last.get('generation', 0)}",
        f"- Final step: {last.get('step', 0)}",
        "",
        "## Summary statistics",
        "",
        "| metric | average | final |",
        "|---|---|---|",
    ]
    for key in [
        "embodied_alive_frac",
        "avida_alive_frac",
        "embodied_mean_lineage_depth",
        "embodied_max_lineage_depth",
        "action_entropy",
        "message_energy",
        "comm_usage_rate",
        "enrichment_separation",
        "coordinated_behavior_index",
        "mean_enrichment",
        "mean_concentration",
        "mean_metabolite",
        "mean_avida_merit",
    ]:
        lines.append(f"| {key} | {avg(key):.4f} | {last.get(key, float('nan')):.4f} |")

    lines.extend(
        [
            "",
            "## Plots",
            "",
            f"![world_fields]({field_plot.relative_to(run_dir)})",
            "",
            f"![agents]({agents_plot.relative_to(run_dir)})",
            "",
            f"![metrics]({metrics_plot.relative_to(run_dir)})",
            "",
            f"![map_elites]({me_plot.relative_to(run_dir)})",
            "",
        ]
    )

    out_path = run_dir / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
