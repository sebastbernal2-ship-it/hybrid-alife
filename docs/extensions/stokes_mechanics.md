# Stokes body mechanics (`EmbodiedConfig.stokes_mechanics`)

Actions become thrust against viscous drag, and drag grows with body radius.

- Swim speed is `thrust / radius` (Stokes drag is proportional to radius); advection by the
  local flow (`flow_movement_scale`) is added on top. Radius comes from the evolvable
  `body_radius` gene, clipped to [0.6, 1.8], when `body_morphology` is on; otherwise it is 1,
  and the flag only switches the move cost to the quadratic form below.
- Move cost is the dissipated swimming power, `cost_move_scale * radius * swim_speed^2 / 0.05`,
  normalised so radius 1 at the full 0.05 step costs what the legacy linear cost does. Advection
  is free (it is not the organism's work).
- `previous_thrust` is the swim velocity, so `previous_velocity - previous_thrust` is still
  exactly the advecting flow that the flow-head policy estimates (checked in
  `tests/test_stokes_mechanics.py`, with the reward terms zeroed so costs are isolated).
- Equal thrust moves a radius-1.8 body one third as fast as a radius-0.6 body; equal speed
  costs radius times more. `body_aspect` has no effect unless `aspect_drag` is on.
- Run: `configs/ablation_stokes_bodies.yaml`.

## Anisotropic drag (`EmbodiedConfig.aspect_drag`)

Needs `stokes_mechanics` and `body_morphology`. The body is a spheroid with the volume of a
sphere of radius `body_radius`; `body_aspect` (clipped to [0.7, 1.4]) is axial length over
equatorial width. Drag along and across the symmetry axis uses the exact Stokes resistance
of Oberbeck (1876) and Perrin (1934) as tabulated by Kim & Karrila, *Microhydrodynamics*
(1991), Tables 3.4 and 3.5 (`physics/spheroid_drag.py`).

- Swim velocity is thrust resolved on the body axis, each part divided by
  `radius * factor`. Cost is force dot velocity. Aspect 1 is bit-identical to the radius law.
- Orientation follows thrust with a one-step lag: the axis is the previous nonzero thrust
  direction, so only a turn pays the anisotropy. A body with no axis yet is aligned with its
  first thrust; newborns and selection replacements start without one. There is no
  rotational drag, Brownian rotation or reorientation by flow shear (no Jeffery orbits).
- At these aspects the two factors differ by less than about 10%.
- Passive advection and the `lbm_live` force coupling are unchanged (isotropic, radius only).
- Tests (`tests/test_spheroid_drag.py`) check the float32 factors against float64, the
  slender-body (Batchelor 1970; Cox 1970) and thin-disk limits, and the exact reduction.
