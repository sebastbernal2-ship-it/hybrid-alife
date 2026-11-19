# Native Avida VM (`AvidaConfig.native`)

`native/avida.cpp` runs each organism's merit-budgeted cycles for one update in C++ (organisms
are independent within an update, so the loop is organism-major). `step_avida_population`
routes through it when `native: true`; division and metabolite coupling stay in JAX on this
branch (`feat/native-avida-full` moves them too).

- Equivalence: with `point_mutation_prob: 0` every state field matches the JAX `vm_cycle`
  loop exactly (merit within float tolerance); `tests/test_avida_native.py` checks it on
  random genomes that exercise all opcodes. With mutation on, copy mutations use a native
  splitmix64 stream, so only the distribution matches JAX.
- Speed: about 12x faster per update at 1024 organisms (0.016 s against 0.19 s).
- Guards: `native=True` under `jax.jit`/`vmap`/`grad` raises `RuntimeError` (the POET
  jit-compiles the Avida step, so use `native=False` there). The bridge rejects genome lengths
  above `max_genome_length` and fewer than 2 registers (the kernel would index out of bounds),
  and refuses to load a kernel whose opcode count differs from the Python VM.
- Run: `configs/ablation_native_avida.yaml`.
