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
- Speed (`benchmarks/bench_avida_native.py`, CPU, median ms per update, 4 cores on a
  shared machine, so expect noise). Native is not automatically faster:

  | workload, organisms | jit JAX | jit native | eager JAX | eager native |
  |---|---|---|---|---|
  | POET config, 8 | 0.28 | 0.89 | 49 | 3.7 |
  | POET config, 4096 | 0.81 | 1.55 | 49 | 4.8 |
  | heavy VM, 8 | 0.76 | 0.90 | 214 | 3.6 |
  | heavy VM, 256 | 1.62 | 1.11 | 210 | 3.4 |
  | heavy VM, 4096 | 7.38 | 3.54 | 234 | 5.3 |

  Eager (the experiment runner) gains over 10x. Under jit the callback has a fixed cost, so on
  the POET config compiled JAX wins at every size measured, and a 16-step POET rollout was
  slower with native (145 ms against 102 ms). Keep `native: false` for stock POET runs. The
  heavy VM (32 max cycles, genome capacity 128) gains up to about 2x from 64 organisms up.
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

## Division quirk

By default, JAX `apply_h_divide` rows with no parent/child match scatter to index 0. When there
are fewer matches than organisms, this undoes a slot-0 child write or parent buffer reset.
`AvidaConfig.divide_scatter_fix=True` sends unmatched rows out of bounds with dropped scatters
in JAX and skips the matching restores in native code. The default remains `False` to preserve
legacy results. The native bridge checks its ABI version and requires `make -B -C native` to
rebuild an incompatible shared library.

How often it fires (JAX path, seeds 7-9): in `base.yaml` (48 updates) and `smoke200.yaml`
(800 updates, one seed) no digital organism dies, so no birth is ever matched and it never
fires. In `ablation_avida_energy.yaml` with `native: false` it fired in 36 of 192 updates (36
of the 47 with a matched birth): 32 of 84 matched offspring were lost, and a slot-0 parent
buffer reset was undone 4 times. These count the legacy path's own trajectory.

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
