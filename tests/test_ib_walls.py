from __future__ import annotations

import json

import numpy as np
import pytest

from hybrid_alife.physics import ib


def _point(y=10.0):
    return ib.Body(np.array([[10.0, y]]), "point")


def test_empty_surfaces_are_bit_identical_and_validation():
    plain = ib.Structure.from_bodies([_point()])
    empty = ib.Structure.from_bodies([_point()], planes=np.empty((0, 3)), discs=np.empty((0, 3)))
    a = ib.ImmersedSim.create(plain, 24, 24).advance(4)
    b = ib.ImmersedSim.create(empty, 24, 24).advance(4)
    np.testing.assert_array_equal(a.X, b.X)
    np.testing.assert_array_equal(a.f, b.f)
    assert b.report["min_wall_gap"] == np.inf

    for kw in (
        {"planes": [(2.0, 0.0, 1.0)], "wall_range": 2.0, "wall_k": 0.1},
        {"discs": [(1.0, 2.0, 0.0)], "wall_range": 2.0, "wall_k": 0.1},
        {"planes": [(1.0, 0.0, 1.0)], "wall_range": 0.0, "wall_k": 0.1},
        {"discs": [(1.0, np.nan, 2.0)], "wall_range": 2.0, "wall_k": 0.1},
    ):
        with pytest.raises(ValueError):
            ib.Structure.from_bodies([_point()], **kw)


def test_plane_pushes_into_fluid_and_reports_true_gap(tmp_path):
    s = ib.Structure.from_bodies(
        [_point(4.0)], planes=[(0.0, 1.0, 2.0)], wall_range=4.0, wall_k=0.05
    )
    out = ib.ImmersedSim.create(s, 24, 24).advance(1)
    assert out.X[0, 1] > 4.0
    assert out.report["min_wall_gap"] == pytest.approx(2.0)
    stronger = ib.Structure.from_bodies(
        [_point(4.0)], planes=[(0.0, 1.0, 2.0)], wall_range=4.0, wall_k=0.1
    )
    stronger_out = ib.ImmersedSim.create(stronger, 24, 24).advance(1)
    assert stronger_out.X[0, 1] - 4.0 == pytest.approx(2 * (out.X[0, 1] - 4.0), rel=2e-3)
    path = tmp_path / "wall.jsonl"
    ib.record_states(ib.ImmersedSim.create(s, 24, 24), path, 0, 1)
    assert json.loads(path.read_text().splitlines()[0])["surfaces"] == {
        "planes": [[0.0, 1.0, 2.0]], "discs": [], "wall_range": 4.0, "wall_k": 0.05
    }


def test_disc_uses_periodic_minimum_image_for_unwrapped_points():
    s = ib.Structure.from_bodies(
        [ib.Body(np.array([[1.0, 12.0]]), "point")],
        discs=[(23.0, 12.0, 3.0)], wall_range=4.0, wall_k=0.05
    )
    out = ib.ImmersedSim.create(s, 24, 24).advance(1)
    # Point x=1 is 22 cells away directly, but 2 cells from the periodic image at x=-1.
    assert out.report["min_wall_gap"] == pytest.approx(-1.0)
    assert out.X[0, 0] > 1.0


def test_soft_plane_repulsion_keeps_stencil_clear_of_masked_wall():
    mask = np.zeros((24, 32), np.uint8)
    mask[[0, -1], :] = 1
    body = ib.Body(np.array([[16.0, 5.0]]), "point")
    no_wall = ib.Structure.from_bodies([body])
    with_wall = ib.Structure.from_bodies(
        [body], planes=[(0.0, 1.0, 2.0)], wall_range=5.0, wall_k=0.2
    )
    pushed = ib.ImmersedSim.create(no_wall, 24, 32, mask=mask, body_force=(0.0, -0.03))
    with pytest.raises(ValueError, match="solid"):
        pushed.advance(100)
    held = ib.ImmersedSim.create(with_wall, 24, 32, mask=mask, body_force=(0.0, -0.03))
    out = held.advance(100)
    assert out.report["min_wall_gap"] > 0.0
