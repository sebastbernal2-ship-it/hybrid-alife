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
  costs radius times more. `body_aspect` has no physical effect yet.
- Run: `configs/ablation_stokes_bodies.yaml`.
