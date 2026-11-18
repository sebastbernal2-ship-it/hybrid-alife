import numpy as np

from hybrid_alife.experiments.latent_law_embodied import (
    lbm_law_profile,
    rotating_geometry,
    rotating_vectors,
)

SIZE = 14


def test_rotated_geometry_moves_walls_with_the_law():
    build = rotating_geometry(lbm_law_profile(SIZE))
    along_r = build(np.array([1.0, 0.0]))
    along_c = build(np.array([0.0, 1.0]))
    # Channel along axis 0: walls are the first/last columns. Along axis 1: rows.
    assert np.all(along_r[:, 0] == 0) and np.all(along_r[:, -1] == 0)
    assert np.all(along_c[0, :] == 0) and np.all(along_c[-1, :] == 0)
    assert np.abs(along_r[:, SIZE // 2]).max() > 0.3 and np.abs(along_c[SIZE // 2]).max() > 0.3


def test_cardinal_rotation_is_exact_quarter_turn():
    profile = lbm_law_profile(SIZE)
    build = rotating_geometry(profile)
    np.testing.assert_allclose(build(np.array([1.0, 0.0])), profile, atol=1e-12)
    turned = build(np.array([0.0, 1.0]))
    rot = np.array([[0.0, -1.0], [1.0, 0.0]])  # maps (1, 0) onto (0, 1)
    for r in range(SIZE):
        for c in range(SIZE):
            np.testing.assert_allclose(
                turned[r, c], rot @ profile[c, SIZE - 1 - r], atol=1e-9, err_msg=f"{r},{c}"
            )


def test_diagonal_field_follows_the_law_direction_and_vectors_variant_differs():
    unit = np.array([1.0, 1.0]) / np.sqrt(2)
    profile = lbm_law_profile(SIZE)
    field = rotating_geometry(profile)(unit)
    speed = np.linalg.norm(field, axis=-1)
    strong = speed > 0.3
    cos = (field[strong] @ unit) / speed[strong]
    assert strong.sum() > 20 and np.median(cos) > 0.9
    assert not np.allclose(field, rotating_vectors(profile)(unit))
