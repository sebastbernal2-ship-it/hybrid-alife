#!/usr/bin/env python
"""V1 compute-scaling sweep for hybrid-alife.

For each generation budget in ``--budgets``, run the same fixed-world config
with that many generations and record the final metric values. Then fit
``metric ~ slope * log10(generations) + intercept`` per metric.

This is a deliberately minimal "compute scaling" — see docs/poet_transfer.md
for the v1 limitations (no power-law CIs, no multi-seed averaging in v1).

Outputs (under --out-dir):

    scaling_slopes.json
    scaling_slopes.md
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from hybrid_alife.experiments.runner import load_config, run_experiment
from hybrid_alife.logging.jsonl import read_jsonl


DEFAULT_METRICS = [
    "action_entropy",
    "mean_avida_merit",
    "qd_score",
    "embodied_alive_frac",
    "avida_tasks_solved",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument(
        "--budgets",
        nargs="+",
        type=int,
        required=True,
        help="Generation counts to sweep, e.g. 10 20 40.",
    )
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/compute_scaling")
    return p.parse_args()


def _run_budget(cfg_path: str, budget: int, seed: int, out_dir: Path) -> dict[str, float]:
    cfg = load_config(cfg_path)
    run_name = f"{cfg.run_name}_b{budget}_s{seed}"
    new_evo = replace(cfg.evolution, generations=budget)
    cfg = replace(
        cfg, seed=seed, run_name=run_name, output_dir=str(out_dir), evolution=new_evo
    )
    run_experiment(cfg)
    metrics_path = out_dir / run_name / "metrics.jsonl"
    rows = read_jsonl(metrics_path) if metrics_path.exists() else []
    final = rows[-1] if rows else {}
    out: dict[str, float] = {}
    for k, v in final.items():
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue
    return out


def least_squares_slope(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Plain least-squares slope of ys vs xs. Returns 0 if degenerate."""
    n = len(xs)
    if n < 2:
        return 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den = sum((x - mean_x) ** 2 for x in xs)
    if den == 0.0:
        return 0.0
    return num / den


def fit_slopes(
    budgets: Sequence[int], finals: Sequence[dict[str, float]], metrics: Sequence[str]
) -> dict:
    log_budgets = [math.log10(max(1, b)) for b in budgets]
    out: dict[str, dict[str, object]] = {}
    for m in metrics:
        values: list[float] = []
        for row in finals:
            v = row.get(m)
            if v is None:
                values.append(float("nan"))
            else:
                values.append(float(v))
        valid = [(x, y) for x, y in zip(log_budgets, values) if not math.isnan(y)]
        if len(valid) >= 2:
            xs = [p[0] for p in valid]
            ys = [p[1] for p in valid]
            slope = least_squares_slope(xs, ys)
        else:
            slope = 0.0
        out[m] = {
            "values": values,
            "slope_per_log10_gen": slope,
        }
    return {"budgets": list(budgets), "metrics": out}


def write_slopes_markdown(summary: dict, out_path: Path) -> None:
    lines = ["# Compute Scaling Slopes\n", f"Budgets: {summary['budgets']}\n"]
    lines.append("| metric | " + " | ".join(str(b) for b in summary["budgets"])
                 + " | slope/log10(gen) |")
    lines.append("|---|" + "|".join(["---:"] * (len(summary["budgets"]) + 1)) + "|")
    for m, info in summary["metrics"].items():
        cells = []
        for v in info["values"]:
            cells.append("nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.4f}")
        lines.append(f"| {m} | " + " | ".join(cells)
                     + f" | {info['slope_per_log10_gen']:+.4f} |")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    finals: list[dict[str, float]] = []
    for budget in args.budgets:
        finals.append(_run_budget(args.config, budget, args.seed, out_dir))

    summary = fit_slopes(args.budgets, finals, args.metrics)

    json_path = out_dir / "scaling_slopes.json"
    md_path = out_dir / "scaling_slopes.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_slopes_markdown(summary, md_path)
    print(f"Scaling slopes written to {json_path}")
    print(f"Markdown summary written to {md_path}")


if __name__ == "__main__":
    main()
