"""Unit tests for the scientific-depth metrics: Bedau, QD, communication, lineage."""

from __future__ import annotations

import numpy as np
import pytest

from hybrid_alife.evolution.archives import MapElitesArchive
from hybrid_alife.experiments.transfer import (
    AblationResult,
    aggregate,
    bootstrap_ci,
    cliffs_delta,
    median_iqr,
    pairwise_effects,
)
from hybrid_alife.metrics.bedau import (
    ActivityTracker,
    adaptive_activity,
    hill_number_1d,
    lineage_hill1d,
)
from hybrid_alife.metrics.communication import (
    bosdis,
    channel_capacity,
    comm_summary,
    posdis,
    shuffle_channel,
    topsim,
    zero_channel,
)
from hybrid_alife.metrics.lineage import LineageTree
from hybrid_alife.metrics.qd import (
    archive_entropy,
    coverage,
    occupancy_entropy,
    qd_score,
    qd_summary,
)


# ---------------------------------------------------------------------------
# Bedau activity
# ---------------------------------------------------------------------------


def test_activity_tracker_basic():
    t = ActivityTracker(persistence_threshold=2)
    t.observe(0, {"a": 1.0})
    s0 = t.stats(0)
    assert s0["A_new"] == 1.0
    assert s0["A_cum"] == 1.0
    assert s0["A_p"] == 0.0  # not yet persistent
    t.observe(1, {"a": 1.0, "b": 1.0})
    t.observe(2, {"a": 1.0})
    s2 = t.stats(2)
    assert s2["A_new"] == 0.0
    assert s2["A_p"] == 1.0  # a persisted >= 2 generations and is current
    assert s2["A_cum"] == 4.0


def test_adaptive_activity_subtracts_shadow():
    exp = [{"A_new": 5.0, "A_cum": 10.0, "A_p": 1.0}]
    shadow = [{"A_new": 2.0, "A_cum": 4.0, "A_p": 0.0}]
    out = adaptive_activity(exp, shadow)
    assert out[0]["adaptive_A_new"] == 3.0
    assert out[0]["adaptive_A_cum"] == 6.0
    assert out[0]["adaptive_A_p"] == 1.0


def test_hill_number_1d_recovers_richness_for_uniform():
    uniform = np.array([10, 10, 10, 10])
    assert hill_number_1d(uniform) == pytest.approx(4.0, rel=1e-6)
    skewed = np.array([100, 1, 1, 1])
    assert hill_number_1d(skewed) < 2.0


def test_lineage_hill1d_with_alive_mask():
    ids = np.array([0, 0, 1, 1, 2])
    alive = np.array([True, True, True, False, True])
    h = lineage_hill1d(ids, alive)
    # 2 lineage-0, 1 lineage-1, 1 lineage-2 alive  -> 2.something
    assert 2.0 < h <= 3.0


# ---------------------------------------------------------------------------
# QD / MAP-Elites
# ---------------------------------------------------------------------------


def test_qd_summary_on_synthetic_archive():
    arch = MapElitesArchive(bins=4)
    rng = np.random.default_rng(0)
    desc = rng.uniform(0, 1, size=(20, 2)).astype(np.float32)
    fit = rng.uniform(0, 1, size=(20,)).astype(np.float32)
    arch.update(desc, fit)
    s = qd_summary(arch)
    assert s["coverage"] > 0.0
    assert s["qd_score"] >= 0.0
    assert s["archive_entropy"] >= 0.0
    assert s["occupancy_entropy"] >= 0.0


def test_qd_score_grows_with_better_fitness():
    arch = MapElitesArchive(bins=4)
    desc = np.array([[0.1, 0.1], [0.5, 0.5]], dtype=np.float32)
    arch.update(desc, np.array([0.0, 0.0], dtype=np.float32))
    low = qd_score(arch)
    arch.update(desc, np.array([5.0, 5.0], dtype=np.float32))
    high = qd_score(arch)
    assert high >= low


def test_archive_entropy_zero_for_single_cell():
    arch = MapElitesArchive(bins=4)
    arch.update(np.array([[0.1, 0.1]], dtype=np.float32), np.array([1.0], dtype=np.float32))
    # one filled cell -> p=1.0 -> entropy=0
    assert occupancy_entropy(arch) == 0.0
    # archive entropy operates on shifted fitness; for one cell still 0
    assert archive_entropy(arch) == 0.0


# ---------------------------------------------------------------------------
# Communication
# ---------------------------------------------------------------------------


def test_topsim_high_when_messages_equal_meanings():
    rng = np.random.default_rng(0)
    meanings = rng.integers(0, 4, size=(80, 3))
    messages = meanings.copy()
    assert topsim(meanings, messages) > 0.9


def test_topsim_low_under_channel_shuffle():
    rng = np.random.default_rng(0)
    meanings = rng.integers(0, 4, size=(80, 3))
    messages = meanings.copy()
    shuffled = shuffle_channel(messages, rng)
    assert topsim(meanings, shuffled) < topsim(meanings, messages)


def test_posdis_bosdis_nonnegative():
    rng = np.random.default_rng(1)
    meanings = rng.integers(0, 4, size=(60, 2))
    messages = meanings.copy()
    assert posdis(meanings, messages) >= 0.0
    assert bosdis(meanings, messages) >= 0.0


def test_channel_capacity_zero_when_zeroed():
    rng = np.random.default_rng(2)
    meanings = rng.integers(0, 4, size=(60, 2))
    messages = meanings.copy()
    z = zero_channel(messages)
    assert channel_capacity(z, meanings[:, 0]) == pytest.approx(0.0, abs=1e-6)


def test_comm_summary_keys():
    rng = np.random.default_rng(3)
    meanings = rng.integers(0, 3, size=(40, 2))
    messages = meanings.copy()
    s = comm_summary(meanings, messages)
    assert set(s) == {"topsim", "posdis", "bosdis"}


# ---------------------------------------------------------------------------
# Lineage tree
# ---------------------------------------------------------------------------


def test_lineage_tree_depth_and_export(tmp_path):
    tree = LineageTree()
    tree.observe(0, np.array([1, 2]), np.array([0, 0]))
    tree.observe(1, np.array([3, 4]), np.array([1, 2]))
    tree.observe(2, np.array([5]), np.array([3]))
    assert tree.depth(5) == 3  # 5 -> 3 -> 1 -> 0
    out = tmp_path / "tree.json"
    tree.export_json(out)
    assert out.exists() and out.stat().st_size > 0
    summary = tree.summary(np.array([5]))
    assert summary["surviving_lineages"] == 1
    assert summary["max_lineage_depth"] == 3


# ---------------------------------------------------------------------------
# Transfer suite / statistics
# ---------------------------------------------------------------------------


def test_median_iqr_and_bootstrap():
    a = np.arange(100, dtype=np.float64)
    med, q25, q75 = median_iqr(a)
    assert med == pytest.approx(49.5, abs=0.6)
    lo, hi = bootstrap_ci(a, n_resample=200, seed=0)
    assert lo <= med <= hi


def test_cliffs_delta_signs():
    a = np.array([1, 2, 3, 4, 5], dtype=np.float64)
    b = np.array([0, 0, 0, 0, 0], dtype=np.float64)
    assert cliffs_delta(a, b) == 1.0
    assert cliffs_delta(b, a) == -1.0
    assert abs(cliffs_delta(a, a)) < 1e-9


def test_aggregate_and_pairwise_effects():
    results = [
        AblationResult(name="base", seed=i, metrics={"score": float(i)}) for i in range(5)
    ] + [
        AblationResult(name="abl", seed=i, metrics={"score": float(i + 10)}) for i in range(5)
    ]
    agg = aggregate(results, "score")
    assert agg["base"]["n"] == 5 and agg["abl"]["median"] > agg["base"]["median"]
    effs = pairwise_effects(results, "score", baseline="base")
    assert effs["abl"] == 1.0
