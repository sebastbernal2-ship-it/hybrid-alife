#!/usr/bin/env python
"""Aggregate PPO-POET summaries and render publication-ready figures."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    comparisons = np.subtract.outer(left, right)
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir")
    parser.add_argument("--out-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.campaign_dir)
    out = Path(args.out_dir) if args.out_dir else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summaries = []
    for path in sorted(root.glob("*/seed-*/summary.json")):
        summaries.append(json.loads(path.read_text(encoding="utf-8")))
    if not summaries:
        raise SystemExit(f"no summary.json files found below {root}")
    grouped: dict[str, list[dict]] = defaultdict(list)
    for summary in summaries:
        grouped[summary["track"]].append(summary)
    table = {}
    for track, rows in sorted(grouped.items()):
        values = np.asarray([row["final_transfer_mean"] for row in rows], dtype=float)
        table[track] = {
            "n": int(values.size),
            "seeds": [int(row["seed"]) for row in rows],
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "iqr": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
            "values": values.tolist(),
        }
    tracks = sorted(table)
    if len(tracks) == 2:
        table["comparison"] = {
            "left": tracks[0],
            "right": tracks[1],
            "cliffs_delta": cliffs_delta(
                np.asarray(table[tracks[0]]["values"]), np.asarray(table[tracks[1]]["values"])
            ),
        }
    result = {"source": str(root), "tracks": table, "summary_count": len(summaries)}
    (out / "statistical_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 4))
    for track, rows in sorted(grouped.items()):
        trajectories = []
        for row in rows:
            values = [item["transfer_score"] for item in row["metrics"]]
            width = max((len(values) // max(int(row["generations"]), 1)), 1)
            trajectories.append(
                [float(np.mean(values[i : i + width])) for i in range(0, len(values), width)]
            )
        length = min(map(len, trajectories))
        mean_trajectory = np.mean([trajectory[:length] for trajectory in trajectories], axis=0)
        axis.plot(np.arange(length), mean_trajectory, label=track)
    axis.set(xlabel="generation", ylabel="transfer score", title="Paired PPO-POET campaign")
    axis.legend()
    figure.tight_layout()
    figure.savefig(out / "transfer_by_track.png", dpi=200)
    plt.close(figure)
    print(out / "statistical_summary.json")


if __name__ == "__main__":
    main()
