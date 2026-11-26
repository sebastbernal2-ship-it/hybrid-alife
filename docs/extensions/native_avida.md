# Native Avida VM (`AvidaConfig.native`)

`native/avida.cpp` runs each organism's merit-budgeted cycles for one update in C++ (organisms
are independent within an update, so the loop is organism-major). `step_avida_population`
routes through it when `native: true`. On this branch the whole update is native: cycles
(`avida_run_update`), division (`avida_divide`, a port of `apply_h_divide`) and the metabolite
uptake/deposit coupling (`avida_couple`).

- Equivalence: with `point_mutation_prob: 0` every state field matches the JAX `vm_cycle`
  loop exactly (merit within float tolerance); `tests/test_avida_native.py` checks it on
  random genomes that exercise all opcodes. With mutation on, copy mutations use a native
  splitmix64 stream, so only the distribution matches JAX.
- Speed: a full native update at 1024 organisms takes about 0.004 s against 0.27-0.32 s for JAX
  (cycles only: 0.016 s against 0.19 s).
- Under `jax.jit` (as the POET compiles the step) and inside `lax.scan`, each update is one
  `jax.pure_callback` into the same host function eager calls use. Budgets, environment inputs
  and int32 kernel seeds come from XLA, so jitted and eager native runs agree, mutation
  included. `vmap` runs the batch sequentially; there are no gradients. The callback copies
  every buffer and calls only NumPy and C++ (JAX calls inside a callback can deadlock).
- Guards: the bridge rejects genome lengths above `max_genome_length` and fewer than 2
  registers (the kernel would index out of bounds), and refuses to load a kernel whose opcode
  count differs from the Python VM. Under jit these checks surface as JAX runtime errors.
  The cycle-only `run_update` entry point is eager-only.
- Run: `configs/ablation_native_avida.yaml`.

## Division quirk (kept on purpose)

In JAX `apply_h_divide`, rows with no parent/child match all scatter to index 0, so when there
are fewer matches than organisms the slot-0 child write and the slot-0 parent buffer reset are
undone. `avida_divide` reproduces this so the two paths stay bit-exact (a test fails without
it). It is a latent JAX bug; fixing it changes legacy behaviour and must be done in both paths.

## Energy and heat (`instruction_energy_cost > 0`)

Each executed instruction costs energy; merit gained converts to energy at
`energy_efficiency`; an organism with no energy stops executing and dies at the end of the
update; a dividing parent hands half its energy to a child that actually survives. Paid energy
is dissipated as heat. State: per-organism `energy`, scalar `energy_in` and `heat`
(`AvidaPopulationState`, `None` unless enabled), with the ledger
`sum(energy) - sum(energy at start) == energy_in - heat` (float32 tolerance for the totals).
Implemented in both the JAX and native paths and compared in `tests/test_avida_native.py`.
`AvidaConfig` rejects a negative cost, a non-positive `energy_initial` when the economy is on,
and `native` with fewer than 2 registers. Run: `configs/ablation_avida_energy.yaml`.
Heat is not in the metrics output or any world field.
