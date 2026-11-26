# Live two-way LBM (`WorldConfig.flow_solver: lbm_live`)

A stateful fluid advanced inside the simulation. `native/lbm.cpp` adds `lbm_init` and
`lbm_advance` (distributions live in a caller-owned buffer, so `LiveLBM` state is plain numpy:
it pickles and is saved and restored by the checkpoint code; legacy checkpoints start a fresh
warmed-up channel).

- Per world step, `step_world` advances `lbm_substeps` LBM steps and refreshes flow, shear and
  lift. Flow is lattice velocity divided by `lbm_peak_speed`. Flow is in the world frame (see the
  `physics/lbm.py` module docstring): the channel runs along world axis 0 with walls on the first
  and last column, which is how `flow_movement_scale` moves agents. The static `lbm` mode uses
  the proxy-field convention unless `lbm_frame: world`.
- Two-way coupling, when `lbm_agent_drag > 0`: each agent pushes the fluid with its Stokes drag
  (`lbm_agent_drag * radius * swim velocity` relative to the fluid, passive advection removed,
  slip clipped to 0.1 lattice units), deposited at its cell for the next step.
  - `lbm_agent_coupling: monopole` treats thrust as an external push (net force on the fluid;
    unphysical for a swimmer).
  - `dipole` is a force-free swimmer: two equal and opposite point forces `lbm_dipole_length`
    cells apart along the swim direction (net force exactly zero). The length is quantised to
    even cell separations because odd gaps excite an undamped grid-scale mode in D2Q9. Agents
    whose either end is on a solid cell, off a non-toroidal domain, or on the same cell are
    skipped whole so the net force stays zero.
- Tests: steady state matches the one-shot solver and Poiseuille; a force pulse changes total
  momentum by exactly the impulse; Newton's third law; dipole net force zero; 200-step runs stay
  finite; checkpoint round trip (`tests/test_lbm_live.py`, `tests/test_lbm_dipole.py`).
- Guards: `WorldConfig` rejects `lbm_viscosity <= 0`, `lbm_peak_speed` outside (0, 0.3],
  `lbm_substeps < 1`, a negative drag, unknown coupling, and `drifting` or `flow_noise_std`
  (the live fluid replaces the flow each step, so they would be silently ignored). `advance`
  rejects force arrays of the wrong shape (the kernel would over-read) and raises when the
  lattice speed exceeds 0.3 or goes non-finite instead of continuing with garbage.
  A 300-step smoke at drag 2.0 peaks at lattice speed 0.19 (monopole) and 0.13 (dipole).
- Limits: below viscosity about 0.02 the fixed 3000-step warm-up is far from steady (slow, not
  unstable); forces deposited on solid cells are dropped; the drag coefficient is uncalibrated;
  agent displacement from non-swimming causes (clipping, escape reflex) counts as swimming.
- Run: `configs/ablation_lbm_live.yaml`.
