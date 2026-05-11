#!/usr/bin/env python
"""Generate a markdown report from an experiment run directory.

Consumes (any subset of):
- metrics.jsonl
- scaling_slopes.json
- transfer_matrix.json
- map_elites.npz
- novelty_archive.npz
- checkpoint_final.pkl

Produces a concise, continuation-safe `report.md` (and optional plots).
See `docs/RESULTS_REPORT_TEMPLATE.md` for the human-curated superset.

Usage:
    # Default: full report with plots
    python scripts/generate_report.py outputs/runs/smoke200

    # Skip plot generation (faster, no matplotlib deps)
    python scripts/generate_report.py outputs/runs/smoke200 --no-plots

    # Pick a non-default output filename
    python scripts/generate_report.py outputs/runs/smoke200 --out summary.md

    # Compare against a baseline run (per-metric average diff)
    python scripts/generate_report.py outputs/runs/exp42 \\
        --baseline outputs/runs/control

    # Only emit the headline-metrics table (for sweep aggregation)
    python scripts/generate_report.py outputs/runs/smoke200 --headline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from hybrid_alife.logging.jsonl import read_jsonl

HEADLINE_METRICS = [
    "embodied_alive_frac",
    "avida_alive_frac",
    "embodied_mean_lineage_depth",
    "embodied_max_lineage_depth",
    "embodied_lineage_hill1d",
    "avida_lineage_hill1d",
    "avida_tasks_solved",
    "action_entropy",
    "message_energy",
    "comm_usage_rate",
    "enrichment_separation",
    "coordinated_behavior_index",
    "mean_enrichment",
    "mean_concentration",
    "mean_metabolite",
    "mean_avida_merit",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=str, help="Path to the run directory")
    p.add_argument(
        "--out",
        type=str,
        default="report.md",
        help="Report filename, relative to run_dir (default: report.md)",
    )
    p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation (avoids matplotlib import; useful in CI).",
    )
    p.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Optional baseline run dir to compute per-metric Δ.",
    )
    p.add_argument(
        "--headline",
        action="store_true",
        help="Emit only the headline-metrics table (for sweep aggregation).",
    )
    return p.parse_args()


def avg_of(records: list[dict[str, Any]], key: str) -> float:
    vals = [r[key] for r in records if key in r]
    return mean(vals) if vals else float("nan")


def fmt(x: float) -> str:
    if x != x:  # NaN
        return "n/a"
    return f"{x:.4f}"


def maybe_load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"_error": f"could not parse {path.name}: {exc}"}


def maybe_load_npz_summary(path: Path) -> dict | None:
    """Return a small summary dict for an .npz archive, or None if missing.

    Uses numpy lazily so the script still imports without numpy installed
    when --no-plots is set and no .npz artifacts are present.
    """
    if not path.exists():
        return None
    try:
        import numpy as np
    except ImportError:
        return {"_error": "numpy not available"}
    try:
        with np.load(path, allow_pickle=True) as data:
            summary: dict[str, Any] = {"keys": list(data.files)}
            for k in data.files:
                arr = data[k]
                summary[f"{k}_shape"] = list(arr.shape)
                if arr.dtype.kind in "fiu" and arr.size:
                    summary[f"{k}_mean"] = float(np.nanmean(arr))
                    summary[f"{k}_nonzero_frac"] = float(np.mean(arr != 0))
            return summary
    except Exception as exc:  # noqa: BLE001 — archive format may vary
        return {"_error": f"could not read {path.name}: {exc}"}


def render_headline_table(
    records: list[dict[str, Any]],
    baseline: list[dict[str, Any]] | None,
) -> list[str]:
    last = records[-1] if records else {}
    lines = ["| metric | average | final | baseline avg | Δ |", "|---|---|---|---|---|"]
    for key in HEADLINE_METRICS:
        a = avg_of(records, key)
        f = last.get(key, float("nan"))
        if baseline is not None:
            b = avg_of(baseline, key)
            d = a - b if (a == a and b == b) else float("nan")
            lines.append(f"| {key} | {fmt(a)} | {fmt(f)} | {fmt(b)} | {fmt(d)} |")
        else:
            lines.append(f"| {key} | {fmt(a)} | {fmt(f)} | — | — |")
    return lines


def render_scaling_section(scaling: dict | None) -> list[str]:
    if scaling is None:
        return ["## Compute-scaling slopes", "", "_No `scaling_slopes.json` present — skipped._", ""]
    if "_error" in scaling:
        return ["## Compute-scaling slopes", "", f"_{scaling['_error']}_", ""]
    lines = ["## Compute-scaling slopes", "", "| axis | slope | r² | n |", "|---|---|---|---|"]
    items = scaling.get("axes", scaling) if isinstance(scaling, dict) else {}
    if isinstance(items, dict):
        for axis, info in items.items():
            if not isinstance(info, dict):
                continue
            slope = info.get("slope", float("nan"))
            r2 = info.get("r2", info.get("r_squared", float("nan")))
            n = info.get("n", info.get("points", "?"))
            lines.append(f"| {axis} | {fmt(float(slope))} | {fmt(float(r2))} | {n} |")
    lines.append("")
    return lines


def render_transfer_section(matrix: dict | None) -> list[str]:
    if matrix is None:
        return ["## POET-style transfer matrix", "", "_No `transfer_matrix.json` present — skipped._", ""]
    if "_error" in matrix:
        return ["## POET-style transfer matrix", "", f"_{matrix['_error']}_", ""]
    lines = ["## POET-style transfer matrix", ""]
    tasks = matrix.get("tasks") or matrix.get("labels")
    grid = matrix.get("matrix") or matrix.get("values")
    if tasks and grid:
        header = "| source \\ target | " + " | ".join(str(t) for t in tasks) + " |"
        sep = "|---" * (len(tasks) + 1) + "|"
        lines.extend([header, sep])
        for src, row in zip(tasks, grid):
            cells = " | ".join(fmt(float(v)) for v in row)
            lines.append(f"| {src} | {cells} |")
    else:
        lines.append("```json")
        lines.append(json.dumps(matrix, indent=2)[:2000])
        lines.append("```")
    lines.append("")
    return lines


def render_archive_section(
    map_elites_summary: dict | None,
    novelty_summary: dict | None,
) -> list[str]:
    if map_elites_summary is None and novelty_summary is None:
        return []
    lines = ["## QD / novelty archives", ""]
    if map_elites_summary is not None:
        lines.append("**MAP-Elites archive** (`map_elites.npz`):")
        lines.append("")
        if "_error" in map_elites_summary:
            lines.append(f"_{map_elites_summary['_error']}_")
        else:
            for k, v in map_elites_summary.items():
                lines.append(f"- `{k}`: {v}")
        lines.append("")
    if novelty_summary is not None:
        lines.append("**Novelty archive** (`novelty_archive.npz`):")
        lines.append("")
        if "_error" in novelty_summary:
            lines.append(f"_{novelty_summary['_error']}_")
        else:
            for k, v in novelty_summary.items():
                lines.append(f"- `{k}`: {v}")
        lines.append("")
    return lines


def render_plots_section(run_dir: Path, no_plots: bool) -> list[str]:
    if no_plots:
        return ["## Plots", "", "_Plot generation skipped (`--no-plots`)._", ""]
    try:
        from hybrid_alife.viz.plots import (
            plot_agent_positions,
            plot_map_elites,
            plot_metrics,
            plot_world_fields,
        )
    except ImportError as exc:
        return ["## Plots", "", f"_Plot deps unavailable: {exc}_", ""]

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    out: list[str] = ["## Plots", ""]
    plotters = [
        ("world_fields", plot_world_fields, run_dir / "checkpoint_final.pkl"),
        ("agents", plot_agent_positions, run_dir / "checkpoint_final.pkl"),
        ("metrics", plot_metrics, run_dir / "metrics.jsonl"),
        ("map_elites", plot_map_elites, run_dir / "map_elites.npz"),
    ]
    for name, fn, src in plotters:
        if not src.exists():
            out.append(f"_{name}: source `{src.name}` missing — skipped._")
            out.append("")
            continue
        try:
            path = fn(src, plots_dir / f"{name}.png")
            out.append(f"![{name}]({path.relative_to(run_dir)})")
            out.append("")
        except Exception as exc:  # noqa: BLE001 — plot helpers may raise broadly
            out.append(f"_{name}: plot failed ({exc})_")
            out.append("")
    return out


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    metrics_path = run_dir / "metrics.jsonl"
    records = read_jsonl(metrics_path) if metrics_path.exists() else []
    n = len(records)
    last = records[-1] if records else {}

    baseline_records: list[dict[str, Any]] | None = None
    if args.baseline:
        baseline_dir = Path(args.baseline)
        baseline_metrics = baseline_dir / "metrics.jsonl"
        if baseline_metrics.exists():
            baseline_records = read_jsonl(baseline_metrics)
        else:
            print(f"warning: baseline metrics not found at {baseline_metrics}")

    if args.headline:
        lines = [f"# Headline metrics: {run_dir.name}", ""]
        lines.extend(render_headline_table(records, baseline_records))
        out_path = run_dir / args.out
        out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Headline report written to {out_path}")
        return

    scaling = maybe_load_json(run_dir / "scaling_slopes.json")
    transfer = maybe_load_json(run_dir / "transfer_matrix.json")
    me_summary = maybe_load_npz_summary(run_dir / "map_elites.npz")
    nov_summary = maybe_load_npz_summary(run_dir / "novelty_archive.npz")

    lines = [
        f"# Experiment Report: {run_dir.name}",
        "",
        f"- Records logged: {n}",
        f"- Generations reached: {last.get('generation', 0)}",
        f"- Final step: {last.get('step', 0)}",
        "",
        "> Template: `docs/RESULTS_REPORT_TEMPLATE.md` is the human-curated"
        " superset of this auto-generated report.",
        "",
        "## Headline metrics",
        "",
    ]
    lines.extend(render_headline_table(records, baseline_records))
    lines.append("")
    lines.extend(render_scaling_section(scaling))
    lines.extend(render_transfer_section(transfer))
    lines.extend(render_archive_section(me_summary, nov_summary))

    lines.extend(
        [
            "## Validation status",
            "",
            "- [ ] Pre-registered hypothesis (`docs/preregistration_template.md`)",
            "- [ ] Seeds ≥ 10 for headline claims",
            "- [ ] Baseline matched in compute and wallclock",
            "- [ ] Decision rule specified (median + IQR + Cliff's δ)",
            "- [ ] Language guardrails followed (`docs/scientific_validation.md` §7)",
            "",
            "## Scientific caveats",
            "",
            "- This report is descriptive. None of the numbers above constitute",
            "  evidence of *open-ended evolution*, *language*, or *tool use* —",
            "  see `docs/scientific_validation.md` §7 for the language",
            "  guardrails.",
            "- Single-seed numbers are pilots. For headline claims run",
            "  `scripts/run_ablation_matrix.py --seeds 10` and report",
            "  median + IQR + Cliff's δ vs the appropriate baseline.",
            "- The proxy field is an *inductive bias*, not a fluid simulator.",
            "  Treat any 'physics' wording accordingly.",
            "- Compute-scaling fits over fewer than ~5 points are anecdotes,",
            "  not power laws. Report CI95 and r².",
            "",
            "## Next experiments",
            "",
            "_TODO — fill from `docs/RESULTS_REPORT_TEMPLATE.md` §7._",
            "",
        ]
    )
    lines.extend(render_plots_section(run_dir, args.no_plots))

    out_path = run_dir / args.out
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written to {out_path}")


if __name__ == "__main__":
    main()
