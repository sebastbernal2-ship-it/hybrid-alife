# PPO-POET ecology pilot v1

This archive records the final bounded ecology validation pilot.

- Source commit: `209c0cc`.
- Tracks: `poet` and `avida_enabled`.
- Seeds: `0`, `1`, and `2`.
- Training: three generations.
- Held-out environments: four per seed from a separate seed stream.
- Archive: admitted environment lineage and frozen archive-transfer matrices.
- Diagnostics: causal gene checks, ecology events, archive admission, and Avida field feedback.

All three seeds admitted three candidate environments and rejected none under the declared `-10.0` raw-reward lower bound.

This is an engineering pilot.
It does not establish open-ended evolution or consciousness.
The preregistered 10-seed campaign, calibrated admission threshold, and independent replication remain required.

## Note added later

This pilot's policies were not saved and were not re-checked. Later checks of other POET runs found
constant-action policies and a -10 criterion that rejects nothing; see
[`../poet_physics_envs/README.md`](../poet_physics_envs/README.md).
