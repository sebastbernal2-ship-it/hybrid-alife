from __future__ import annotations

import numpy as np

from scripts.analyze_poet_campaign import assess_success, paired_comparison


def test_paired_comparison_reports_campaign_statistics() -> None:
    result = paired_comparison(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0]))
    assert result["n_pairs"] == 2
    assert result["mean_difference"] == 1.5
    assert result["wins"] == 2
    assert result["ties"] == 0
    assert result["exact_paired_sign_permutation_p"] == 0.5
    assert result["cliffs_delta"] == 0.75


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
