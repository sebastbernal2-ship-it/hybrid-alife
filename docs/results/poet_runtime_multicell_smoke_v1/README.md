# PPO-POET multi-cell timing smoke v1

This bounded smoke checks that the compiled frozen-rollout evaluator runs across multiple seeds and both comparison tracks.

- Source commit: `88f6203`.
- Tracks: `poet` and `static`.
- Seeds: `0` and `1`.
- Generations: `20`.
- Elapsed time: `98.40` seconds.
- Cache declaration: cleared.

All four cells completed and the analyzer produced a statistical summary.
The two-seed exploratory values were poet mean `7.736` and static mean `3.113`.
The paired exact sign-permutation p-value was `0.5`.
This smoke is not a scientific comparison and does not satisfy the preregistered replication criteria.
It only confirms bounded multi-cell execution after the evaluator optimization.
