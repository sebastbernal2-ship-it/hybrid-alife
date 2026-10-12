# PPO-POET runtime profile v1

This record bounds the local CPU timing path before another preregistered campaign.

- Configuration: `configs/poet_campaign.yaml`.
- Tracks: `poet` and `static`.
- Seeds: one seed, `0`.
- Machine: local CPU.
- Archive selection limit: `16`.
- Causal validity limit: `16`.

## Measurements

- Before the compiled evaluator, two generations per track completed in `37.57` seconds.
- Before the compiled evaluator, twenty generations per track completed in `2:23.34`.
- After the compiled evaluator, two generations per track completed in `34.77` seconds including compilation.
- After the compiled evaluator, twenty generations per track completed in `1:19.47`.
- The 200-generation profiled rerun completed one first cell in about `21` minutes before it was stopped.

The compiled evaluator reduced the repeated frozen-rollout path for the bounded smoke.
The full 40-cell campaign still requires a fresh multi-cell timing check before launch.
The retained fast path applies only to deterministic no-Avida evaluation.
A JAX `vmap` transfer-matrix variant was tested and rejected because its 50-generation smoke took `3:36`, slower than the retained per-cell compiled path.

These timings are engineering evidence only.
No PPO-POET scientific result is attached to this profile.
