#!/usr/bin/env python
"""Aggregate PPO-POET summaries and render publication-ready figures."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CURRENT_SCHEMA_VERSION = 2
_REQUIRED_SUMMARY_FIELDS = {
    "schema_version",
    "track",
    "seed",
    "final_transfer_mean",
    "heldout_transfer_mean",
    "source_commit",
    "replication_metadata",
}
_REQUIRED_REPLICATION_FIELDS = {
    "replication_id",
    "seed_offset",
    "operator",
    "machine_label",
    "cache_cleared",
}


def discover_summary_paths(root: Path) -> list[Path]:
    """Find summary artifacts in both flat and replication-nested layouts."""
    return sorted(root.rglob("summary.json"))


def _campaign_cell_key(summary: dict, path: Path) -> tuple[str, str, int]:
    missing = _REQUIRED_SUMMARY_FIELDS - summary.keys()
    if missing:
        raise ValueError(f"{path}: missing summary fields: {sorted(missing)}")
    if summary["schema_version"] != CURRENT_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: unsupported schema_version {summary['schema_version']!r}; "
            f"expected {CURRENT_SCHEMA_VERSION}"
        )
    metadata = summary["replication_metadata"]
    if not isinstance(metadata, dict):
        raise ValueError(f"{path}: replication_metadata must be an object")
    missing = _REQUIRED_REPLICATION_FIELDS - metadata.keys()
    if missing:
        raise ValueError(f"{path}: missing replication metadata: {sorted(missing)}")
    return (str(metadata["replication_id"]), str(summary["track"]), int(summary["seed"]))


def load_campaign_summaries(root: Path) -> list[dict]:
    """Load validated summaries without allowing duplicate cells to overwrite."""
    summaries = []
    seen: dict[tuple[str, str, int], Path] = {}
    for path in discover_summary_paths(root):
        summary = json.loads(path.read_text(encoding="utf-8"))
        key = _campaign_cell_key(summary, path)
        if key in seen:
            raise ValueError(
                f"duplicate campaign cell {key} in {seen[key]} and {path}"
            )
        seen[key] = path
        summaries.append(summary)
    if not summaries:
        raise SystemExit(f"no summary.json files found below {root}")
    return summaries


def _group_summaries(summaries: list[dict]) -> dict[str, dict[str, dict[int, dict]]]:
    grouped: dict[str, dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    seen: dict[tuple[str, str, int], int] = {}
    for index, summary in enumerate(summaries):
        metadata = summary.get("replication_metadata", {})
        key = (
            str(metadata.get("replication_id", "unknown")),
            str(summary.get("track")),
            int(summary["seed"]),
        )
        if key in seen:
            raise ValueError(
                f"duplicate campaign cell {key} at summary indexes {seen[key]} and {index}"
            )
        seen[key] = index
        grouped[key[0]][key[1]][key[2]] = summary
    return grouped


def cliffs_delta(left: np.ndarray, right: np.ndarray) -> float:
    comparisons = np.subtract.outer(left, right)
    return float((np.sum(comparisons > 0) - np.sum(comparisons < 0)) / comparisons.size)


def exact_paired_sign_permutation(left: np.ndarray, right: np.ndarray) -> float:
    differences = np.asarray(right, dtype=float) - np.asarray(left, dtype=float)
    if differences.size == 0:
        raise ValueError("paired comparison requires at least one value")
    observed = abs(float(np.mean(differences)))
    exceedances = 0
    for signs in itertools.product((-1.0, 1.0), repeat=differences.size):
        permuted = float(np.mean(differences * np.asarray(signs)))
        if abs(permuted) >= observed - 1e-12:
            exceedances += 1
    return exceedances / (2 ** differences.size)


def bootstrap_mean_ci(differences: np.ndarray, *, samples: int = 10_000) -> list[float]:
    """Return a deterministic percentile bootstrap interval for paired effects."""
    if differences.size == 0:
        raise ValueError("bootstrap requires at least one difference")
    rng = np.random.default_rng(0)
    draws = rng.choice(differences, size=(samples, differences.size), replace=True)
    means = np.mean(draws, axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_comparison(left: np.ndarray, right: np.ndarray) -> dict[str, float | int | list[float]]:
    if left.size != right.size:
        raise ValueError("paired comparison requires equal sample sizes")
    differences = right - left
    return {
        "n_pairs": int(differences.size),
        "mean_difference": float(np.mean(differences)),
        "wins": int(np.sum(differences > 0)),
        "ties": int(np.sum(differences == 0)),
        "exact_paired_sign_permutation_p": exact_paired_sign_permutation(left, right),
        "cliffs_delta": cliffs_delta(right, left),
        "bootstrap_ci_95": bootstrap_mean_ci(differences),
    }


def assess_success(
    summaries: list[dict], *, primary_track: str = "poet", baseline_track: str = "static"
) -> dict:
    """Apply the preregistered replicated held-out transfer success criteria."""
    grouped = _group_summaries(summaries)
    per_replication = {}
    all_differences = []
    valid_replications = 0
    for replication_id, tracks in sorted(grouped.items()):
        primary = tracks.get(primary_track, {})
        baseline = tracks.get(baseline_track, {})
        seeds = sorted(set(primary) & set(baseline))
        seeds = [
            seed
            for seed in seeds
            if primary[seed].get("heldout_transfer_mean") is not None
            and baseline[seed].get("heldout_transfer_mean") is not None
        ]
        metadata_rows = [
            summary.get("replication_metadata", {})
            for track in tracks.values()
            for summary in track.values()
        ]
        metadata = metadata_rows[0] if metadata_rows else {}
        metadata_consistent = all(row == metadata for row in metadata_rows)
        if not seeds:
            per_replication[replication_id] = {
                "replication_id": replication_id,
                "n_pairs": 0,
                "seed_count_requirement_met": False,
                "metadata_consistent": metadata_consistent,
                "cache_cleared": bool(metadata.get("cache_cleared", False)),
                "seed_offset": metadata.get("seed_offset"),
            }
            continue
        left = np.asarray([baseline[seed]["heldout_transfer_mean"] for seed in seeds], dtype=float)
        right = np.asarray([primary[seed]["heldout_transfer_mean"] for seed in seeds], dtype=float)
        comparison = paired_comparison(left, right)
        comparison.update(
            {
                "replication_id": replication_id,
                "seed_count_requirement_met": len(seeds) >= 10,
                "metadata_consistent": metadata_consistent,
                "cache_cleared": bool(metadata.get("cache_cleared", False)),
                "seed_offset": int(metadata["seed_offset"]),
            }
        )
        per_replication[replication_id] = comparison
        all_differences.extend((right - left).tolist())
        if (
            len(seeds) >= 10
            and metadata_consistent
            and comparison["mean_difference"] > 0
            and comparison["exact_paired_sign_permutation_p"] < 0.05
        ):
            valid_replications += 1
    differences = np.asarray(all_differences, dtype=float)
    ci = bootstrap_mean_ci(differences) if differences.size else [None, None]
    offsets = [item["seed_offset"] for item in per_replication.values()]
    independent_offsets = len(offsets) == len(set(offsets))
    success = (
        len(per_replication) >= 2
        and valid_replications == len(per_replication)
        and bool(differences.size)
        and float(np.mean(differences)) > 0
        and ci[0] is not None
        and ci[0] > 0
        and independent_offsets
        and all(item["metadata_consistent"] for item in per_replication.values())
        and all(item["cache_cleared"] for item in per_replication.values())
    )
    return {
        "status": "success" if success else "insufficient_evidence",
        "primary_track": primary_track,
        "baseline_track": baseline_track,
        "replications": len(per_replication),
        "valid_replications": valid_replications,
        "independent_seed_offsets": independent_offsets,
        "bootstrap_ci_95": ci,
        "mean_difference": float(np.mean(differences)) if differences.size else None,
        "per_replication": per_replication,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--primary-track", default="poet")
    parser.add_argument("--baseline-track", default="static")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.campaign_dir)
    out = Path(args.out_dir) if args.out_dir else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summaries = load_campaign_summaries(root)
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
    comparison_order = [
        name for name in ("isolated", "avida_enabled", "avida_persistent") if name in tracks
    ]
    comparison_order.extend(name for name in tracks if name not in comparison_order)
    comparisons = {}
    for left_name, right_name in itertools.combinations(comparison_order, 2):
        left = np.asarray(table[left_name]["values"])
        right = np.asarray(table[right_name]["values"])
        comparisons[f"{right_name}_minus_{left_name}"] = {
            "left": left_name,
            "right": right_name,
            **paired_comparison(left, right),
        }
    result = {
        "schema_version": CURRENT_SCHEMA_VERSION,
        "source": str(root),
        "tracks": table,
        "comparisons": comparisons,
        "summary_count": len(summaries),
        "source_commits": sorted(
            {summary.get("source_commit", "unknown") for summary in summaries}
        ),
        "success_analysis": assess_success(
            summaries,
            primary_track=args.primary_track,
            baseline_track=args.baseline_track,
        ),
    }
    if len(comparisons) == 1:
        result["comparison"] = next(iter(comparisons.values()))
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
