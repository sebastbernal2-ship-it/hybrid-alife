# LBM flow solver (`WorldConfig.flow_solver: lbm`)

`native/lbm.cpp` is a D2Q9 BGK lattice-Boltzmann solver with a Guo body force, periodic in x and
y, halfway bounce-back at masked solid cells, exposed through a C ABI and
`hybrid_alife.physics.lbm` (ctypes). `initialize_world` solves a steady channel flow once
(walls on the top and bottom rows, circular obstacle when `curvature_regime: obstacle`),
normalised to unit peak speed, and uses it in place of the proxy flow.

- Validation: matches the analytic Poiseuille profile within 2%, conserves mass, no-slip at
  walls (`tests/test_lbm_physics.py`).
- Guards: `solve` raises if the lattice speed exceeds 0.3 (D2Q9 is invalid there even when the
  result is finite), for invalid shapes, and for `nu <= 0` (`tau <= 0.5`). `WorldConfig`
  rejects unknown `flow_solver` values at construction.
- Frame: flow component 0 is the x (column) component, the same convention as the proxy
  fields. Agent movement uses component 0 along the row axis, a pre-existing mismatch in the
  repo; the law-study and live-LBM branches transpose the field where it matters.
- Run: `configs/ablation_lbm_flow.yaml`.
