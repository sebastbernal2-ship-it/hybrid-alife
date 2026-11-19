# Opt-in physics and native extensions

Every extension is off by default, so legacy behaviour and recorded results are unchanged.
Each lives on its own `feat/*` branch; its feature doc (named below) exists on that branch.

| Extension | Flag(s) | Native C++ | Feature doc | Config |
|---|---|---|---|---|
| LBM flow solver | `WorldConfig.flow_solver: lbm` | yes (`native/lbm.cpp`) | `lbm_flow.md` | `configs/ablation_lbm_flow.yaml` |
| Native Avida VM | `AvidaConfig.native` | yes (`native/avida.cpp`) | `native_avida.md` | `configs/ablation_avida_energy.yaml` |
| Avida energy and heat | `AvidaConfig.instruction_energy_cost` | with the above | `native_avida.md` | `configs/ablation_avida_energy.yaml` |
| Thermodynamic eating | `EmbodiedConfig.thermo_efficiency` | no | `energy_matter_conservation.md` | `configs/ablation_thermo.yaml` |
| Chemical signaling | `EmbodiedConfig.chemical_signaling` | no | `chemical_signaling.md` | `configs/ablation_chemical_signaling.yaml` |
| Stokes body mechanics | `EmbodiedConfig.stokes_mechanics` | no | `stokes_mechanics.md` | `configs/ablation_stokes_bodies.yaml` |
| Live two-way LBM | `WorldConfig.flow_solver: lbm_live` | yes | `lbm_live.md` | `configs/ablation_lbm_live.yaml` |

## Native kernels

- Needs `g++` and `make`. Build with `make -C native` (writes `src/hybrid_alife/physics/lib*.so`,
  which are git-ignored). The Python bridges also build on first import; a failed build raises
  a `RuntimeError` with the compiler output instead of an obscure loader error.
- Native code is exercised by the same tests that cover the JAX path. The pattern is a
  bit-exact comparison with the stochastic knobs off (mutation probabilities 0), plus an
  invariants test with them on, because the native RNG matches JAX in distribution only.
- `AvidaConfig.native` is not supported under `jax.jit` (the POET jit-compiles the Avida step);
  it raises a clear error rather than failing inside the tracer.
- Sanitizer check used during review: compile a kernel with
  `-fsanitize=address,undefined -Wall -Wextra` and drive it with a small C++ harness.

## Sharp edges

- LBM is only valid at low lattice speed. The solver raises above 0.3 (lattice units), and at
  viscosity below about 0.02 a fixed warm-up is far from steady (slow, not unstable).
- The dipole swimmer coupling quantises its length to even cell separations (odd gaps excite
  an undamped grid-scale mode in D2Q9).
- `thermo_efficiency` closes eating only. Reward terms, predation, `max_energy` clipping and
  Avida metabolite uptake are outside the matter-plus-energy ledger.
- `chemical_signaling` needs the concentration field to be visible to the agent (not hidden,
  not blind), otherwise the signal channel is silently unobservable; construction rejects it.
