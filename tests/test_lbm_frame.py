"""One frame convention for the LBM modes, and pins on legacy outputs and recorded numbers."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax
import numpy as np
import pytest

from hybrid_alife.experiments.latent_law_embodied import lbm_law_profile
from hybrid_alife.experiments.runner import load_config
from hybrid_alife.world.env import initialize_world, new_live_lbm

RESULTS = Path(__file__).resolve().parents[1] / "docs" / "results"


def _world(**kw):
    base = load_config("configs/ablation_lbm_flow.yaml").world
    return replace(base, width=18, height=12, flow_noise_std=0.0, **kw)


def _flow(cfg):
    return np.asarray(initialize_world(cfg, jax.random.PRNGKey(0)).flow)


def test_legacy_lbm_flow_is_pinned():
    flow = _flow(_world())
    expected = {
        (3, 4): (0.9999653, -0.0349898), (6, 4): (0.0, 0.0),
        (6, 9): (0.4075998, 0.0694944), (8, 2): (0.4346800, 0.2313110),
    }
    for cell, value in expected.items():
        np.testing.assert_allclose(flow[cell], value, atol=1e-5)
    np.testing.assert_allclose(flow.sum(axis=(0, 1)), (77.31385, 0.00051), atol=1e-3)
    assert np.all(flow[0] == 0) and np.all(flow[-1] == 0)  # legacy walls are rows


def test_law_profile_used_by_the_committed_studies_is_pinned():
    profile = lbm_law_profile(14)
    expected = {
        (3, 4): (0.5886060, -0.0994513), (3, 7): (0.0, 0.0), (10, 2): (0.3813634, 0.0147852),
    }
    for cell, value in expected.items():
        np.testing.assert_allclose(profile[cell], value, atol=1e-5)
    np.testing.assert_allclose(profile.sum(axis=(0, 1)), (61.81479, 0.000266), atol=1e-3)


@pytest.mark.parametrize(
    ("path", "passes", "median"),
    [
        ("embodied_law_lbm_live/seeds0-14_A_legacy.json", 14, 0.393),
        ("embodied_law_lbm_live/seeds0-14_E_mono_2.0.json", 10, 0.255),
        ("embodied_law_lbm_profile/seeds0-14_uniform.json", 14, 0.393),
        ("embodied_law_lbm_profile/seeds0-14_lbm_channel_geometry.json", 10, 0.308),
    ],
)
def test_committed_study_numbers_are_pinned(path, passes, median):
    rows = json.loads((RESULTS / path).read_text())
    assert sum(r["passed"] for r in rows) == passes
    assert round(float(np.median([r["min_delta"] for r in rows])), 3) == median


@pytest.mark.parametrize("regime", ["default", "obstacle"])
def test_world_frame_legacy_lbm_matches_the_live_channel(regime):
    """`lbm_frame: world` is the lbm_live convention: component 0 along rows, walls in columns."""
    legacy = _world(curvature_regime=regime)
    world = _flow(replace(legacy, lbm_frame="world"))
    live = new_live_lbm(replace(legacy, flow_solver="lbm_live", lbm_viscosity=0.1)).flow()
    assert np.all(world[:, 0] == 0) and np.all(world[:, -1] == 0)
    assert np.all((np.abs(world).sum(-1) == 0) == (np.abs(live).sum(-1) == 0))
    # Both are unit-peak up to solver warm-up; compare shapes after normalising.
    np.testing.assert_allclose(world / np.abs(world).max(), live / np.abs(live).max(), atol=0.03)


def test_unknown_lbm_frame_is_rejected():
    with pytest.raises(ValueError, match="lbm_frame"):
        _world(lbm_frame="rows")
