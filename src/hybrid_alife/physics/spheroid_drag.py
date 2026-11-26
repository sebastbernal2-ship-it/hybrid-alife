"""Stokes translational drag on a spheroid (``EmbodiedConfig.aspect_drag``).

Exact creeping-flow resistance of Oberbeck (1876) and Perrin (1934), in the form of
Kim & Karrila, "Microhydrodynamics" (1991), resistance functions X^A (along the symmetry
axis) and Y^A (across it) for prolate (Table 3.4) and oblate (Table 3.5) spheroids.

``aspect`` is axial length / equatorial width. Bodies keep the volume of a sphere of radius
r, and the returned factors are drag relative to that sphere (6 pi mu r U), so aspect 1
gives exactly (1, 1). Near the sphere the closed forms cancel catastrophically in float32,
so their series in eccentricity squared is used.
"""

from __future__ import annotations

import jax.numpy as jnp

_SERIES_BELOW = 0.05  # eccentricity squared


def drag_factors(aspect):
    """Return (along_axis, across_axis) drag factors for ``aspect`` > 0."""
    aspect = jnp.asarray(aspect, dtype=jnp.float32)
    prolate = aspect >= 1.0
    e2 = jnp.where(prolate, 1.0 - 1.0 / jnp.maximum(aspect, 1.0) ** 2,
                   1.0 - jnp.minimum(aspect, 1.0) ** 2)
    e2_exact = jnp.maximum(e2, _SERIES_BELOW)  # keeps the unused branch finite
    e = jnp.sqrt(e2_exact)
    root = jnp.sqrt(1.0 - e2_exact)
    e3 = e * e2_exact
    # ln((1 + e) / (1 - e)) written so that slender bodies (e -> 1) do not round to inf.
    one_minus_e = (1.0 - e2_exact) / (1.0 + e)
    log = jnp.log((1.0 + e) / one_minus_e)
    arc = jnp.arctan2(e, root)  # arcsin(e)
    pro_x = (8.0 / 3.0) * e3 / (-2.0 * e + (1.0 + e2_exact) * log)
    pro_y = (16.0 / 3.0) * e3 / (2.0 * e + (3.0 * e2_exact - 1.0) * log)
    obl_x = (4.0 / 3.0) * e3 / ((2.0 * e2_exact - 1.0) * arc + e * root)
    obl_y = (8.0 / 3.0) * e3 / ((2.0 * e2_exact + 1.0) * arc - e * root)
    small = e2 < _SERIES_BELOW
    x = jnp.where(
        small,
        jnp.where(prolate, 1.0 - e2 * (2.0 / 5.0) - e2**2 * (17.0 / 175.0),
                  1.0 - e2 * (1.0 / 10.0) - e2**2 * (31.0 / 1400.0)),
        jnp.where(prolate, pro_x, obl_x),
    )
    y = jnp.where(
        small,
        jnp.where(prolate, 1.0 - e2 * (3.0 / 10.0) - e2**2 * (57.0 / 700.0),
                  1.0 - e2 * (1.0 / 5.0) - e2**2 * (79.0 / 1400.0)),
        jnp.where(prolate, pro_y, obl_y),
    )
    # Reference length of the normalisation above, in units of r: the semi-major axis for
    # prolate (aspect^(2/3)), the equatorial radius for oblate (aspect^(-1/3)).
    length = jnp.where(prolate, aspect ** (2.0 / 3.0), aspect ** (-1.0 / 3.0))
    return x * length, y * length
