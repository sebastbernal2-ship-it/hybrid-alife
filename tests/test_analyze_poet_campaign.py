from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_poet_campaign import (
    assess_success,
    discover_summary_paths,
    load_campaign_summaries,
    paired_comparison,
)


def test_paired_comparison_reports_campaign_statistics() -> None:
    result = paired_comparison(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]))
    assert result["n_pairs"] == 2
    assert result["mean_difference"] == 1.5
    assert result["wins"] == 2
    assert result["ties"] == 0
    assert result["exact_paired_sign_permutation_p"] == 0.5
    assert result["cliffs_delta"] == 0.75


def test_analyzer_discovers_two_replication_layout(tmp_path: Path) -> None:
    for replication_id in ("replication-1", "replication-2"):
        for track in ("poet", "static"):
            path = tmp_path / replication_id / track / "seed-000" / "summary.json"
            path.parent.mkdir(parents=True)
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "track": track,
                        "seed": 0,
                        "final_transfer_mean": 1.0,
                        "heldout_transfer_mean": 1.0,
                        "source_commit": "abc123",
                        "replication_metadata": {
                            "replication_id": replication_id,
                            "seed_offset": 0 if replication_id == "replication-1" else 1000,
                            "operator": "tester",
                            "machine_label": "machine",
                            "cache_cleared": True,
                        },
                    }
                )
            )

    paths = discover_summary_paths(tmp_path)
    assert len(paths) == 4
    summaries = load_campaign_summaries(tmp_path)
    assert len(summaries) == 4


def test_analyzer_rejects_duplicate_replication_track_seed(tmp_path: Path) -> None:
    payload = {
        "schema_version": 2,
        "track": "poet",
        "seed": 0,
        "final_transfer_mean": 1.0,
        "heldout_transfer_mean": 1.0,
        "source_commit": "abc123",
        "replication_metadata": {
            "replication_id": "replication-1",
            "seed_offset": 0,
            "operator": "tester",
            "machine_label": "machine",
            "cache_cleared": True,
        },
    }
    for name in ("first", "second"):
        path = tmp_path / name / "poet" / "seed-000" / "summary.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match="duplicate campaign cell"):
        load_campaign_summaries(tmp_path)


def test_assess_success_requires_two_independent_replications() -> None:
    rows = []
    for replication_id in ("r1", "r2"):
        for seed in range(10):
            rows.extend(
                [
                    {
                        "track": "poet",
                        "seed": seed,
                        "heldout_transfer_mean": 2.0,
                        "replication_metadata": {
                            "replication_id": replication_id,
                            "seed_offset": 0 if replication_id == "r1" else 1000,
                            "operator": replication_id,
                            "machine_label": replication_id,
                            "cache_cleared": True,
                        },
                    },
                    {
                        "track": "static",
                        "seed": seed,
                        "heldout_transfer_mean": 0.0,
                        "replication_metadata": {
                            "replication_id": replication_id,
                            "seed_offset": 0 if replication_id == "r1" else 1000,
                            "operator": replication_id,
                            "machine_label": replication_id,
                            "cache_cleared": True,
                        },
                    },
                ]
            )
    result = assess_success(rows, primary_track="poet", baseline_track="static")
    assert result["status"] == "success"
    assert result["replications"] == 2
    assert result["per_replication"]["r1"]["exact_paired_sign_permutation_p"] < 0.05
    assert result["bootstrap_ci_95"][0] > 0
