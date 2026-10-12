from __future__ import annotations

from hybrid_alife.experiments.field_discovery import run_evaluation, summarize_results


def test_field_discovery_is_reproducible_and_reports_scoped_outcomes():
    first = run_evaluation(seed=7, repetitions=8, episodes=80, intervention_at=40)
    second = run_evaluation(seed=7, repetitions=8, episodes=80, intervention_at=40)

    assert first == second
    assert first["schema_version"] == 1
    assert set(first["access_modes"]) == {"direct", "indirect"}
    for outcome in first["access_modes"].values():
        assert 0.0 <= outcome["field_use_gain"] <= 1.0
        assert 0.0 <= outcome["intervention_adaptation"] <= 1.0
        assert 0.0 <= outcome["pre_intervention_accuracy"] <= 1.0
        assert 0.0 <= outcome["post_intervention_accuracy"] <= 1.0
    assert first["access_modes"]["direct"]["field_use_gain"] > 0
    assert first["access_modes"]["direct"]["intervention_adaptation"] > 0
    direct = first["access_modes"]["direct"]["post_intervention_accuracy"]
    indirect = first["access_modes"]["indirect"]["post_intervention_accuracy"]
    assert direct > indirect


def test_summary_aggregates_runs_without_dropping_access_modes():
    runs = [
        run_evaluation(seed=seed, repetitions=4, episodes=40, intervention_at=20)
        for seed in (2, 5)
    ]
    summary = summarize_results(runs)

    assert summary["run_count"] == 2
    assert set(summary["access_modes"]) == {"direct", "indirect"}
    assert summary["access_modes"]["direct"]["field_use_gain"]["n"] == 2
