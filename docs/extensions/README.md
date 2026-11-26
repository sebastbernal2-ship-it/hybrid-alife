# Opt-in physics and native extensions

Every extension is off by default, so legacy behaviour and recorded results are unchanged.
Each lives on its own `feat/*` branch; its feature doc (named below) exists on that branch.

| Extension | Flag(s) | Native C++ | Feature doc | Config |
|---|---|---|---|---|
| LBM flow solver | `WorldConfig.flow_solver: lbm` | yes (`native/lbm.cpp`) | `lbm_flow.md` | `configs/ablation_lbm_flow.yaml` |
| Native Avida VM | `AvidaConfig.native` | yes (`native/avida.cpp`) | `native_avida.md` | `configs/ablation_native_avida.yaml` |
| Avida energy and heat | `AvidaConfig.instruction_energy_cost` | with the above | `native_avida.md` | `configs/ablation_avida_energy.yaml` |
| Thermodynamic eating | `EmbodiedConfig.thermo_efficiency` | no | `energy_matter_conservation.md` | `configs/ablation_thermo.yaml` |
| Chemical signaling | `EmbodiedConfig.chemical_signaling` | no | `chemical_signaling.md` | `configs/ablation_chemical_signaling.yaml` |
| Stokes body mechanics | `EmbodiedConfig.stokes_mechanics` | no | `stokes_mechanics.md` | `configs/ablation_stokes_bodies.yaml` |
| Anisotropic spheroid drag | `EmbodiedConfig.aspect_drag` (+ `stokes_mechanics`) | no | `stokes_mechanics.md` | - |
| Energy ledger, heat metric, lossy predation | `EmbodiedConfig.energy_ledger`, `predation_efficiency` | no | `energy_matter_conservation.md` | - |
| Multi-channel signal content | `EmbodiedConfig.signal_content`, `WorldConfig.concentration_channels` | no | `chemical_signaling.md` | - |
| Live two-way LBM | `WorldConfig.flow_solver: lbm_live` | yes | `lbm_live.md` | `configs/ablation_lbm_live.yaml` |
| Immersed-boundary bodies and contact | none: import `hybrid_alife.physics.ib` | yes (`native/ib.cpp`, `contact.cpp`) | `immersed_boundary.md` | none (standalone) |

## Native kernels

- Needs `g++` and `make`. Build with `make -C native` (writes `src/hybrid_alife/physics/lib*.so`,
  which are git-ignored). The Python bridges also build on first import; a failed build raises
  a `RuntimeError` with the compiler output instead of an obscure loader error.
- Native code is exercised by the same tests that cover the JAX path. The pattern is a
  bit-exact comparison with the stochastic knobs off (mutation probabilities 0), plus an
  invariants test with them on, because the native RNG matches JAX in distribution only.
- `AvidaConfig.native` works under `jax.jit` (including `lax.scan`): each update is one
  `jax.pure_callback`, and `vmap` runs it sequentially (no gradients). Kernel seeds come from
  the JAX key, so jitted and eager native runs agree. Input-validation errors raised inside the
  callback surface as JAX runtime errors. Under jit it can be slower than compiled JAX (it is
  on the POET config); see `native_avida.md`.
- Sanitizer check used during review: compile a kernel with
  `-fsanitize=address,undefined -Wall -Wextra` and drive it with a small C++ harness.

## Sharp edges

- LBM is only valid at low lattice speed. The solver raises above 0.3 (lattice units), and at
  viscosity below about 0.02 a fixed warm-up is far from steady (slow, not unstable).
- The dipole swimmer coupling quantises its length to even cell separations (odd gaps excite
  an undamped grid-scale mode in D2Q9).
- `thermo_efficiency` closes eating only. Reward terms, predation and `max_energy` clipping
  are tracked by `energy_ledger`; Avida metabolite uptake is outside both.
- `chemical_signaling` needs the concentration field to be visible to the agent (not hidden,
  not blind), otherwise the signal channel is silently unobservable; construction rejects it.
