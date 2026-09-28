# Transfer and Compute-Scaling Semantics

This document defines the transfer and compute-scaling harness for
`hybrid-alife`.

## Transfer modes

`scripts/run_transfer_matrix.py` supports two modes:

- `--mode reeval` runs each source and target configuration independently and
  reports final-metric deltas.
- `--mode transfer` trains each source configuration, loads its final
  checkpoint, and evaluates only the source embodied controller genomes in a
  fresh target world.

Fixed-policy transfer resets target world and agent state. It copies controller
parameters only. Mutation, selection, reproduction, and Avida updates are
disabled during evaluation. Source and target controller tensor shapes must
match. A shape mismatch fails instead of silently changing the policy.

Example:

```bash
python scripts/run_transfer_matrix.py \
  --mode transfer \
  --source-configs configs/transfer_source.yaml \
  --target-configs configs/transfer_source.yaml configs/transfer_target_uniform.yaml \
  --seed 0 \
  --out-dir outputs/transfer
```

The fixed-policy JSON contains `mode: fixed_policy_transfer` and one direct
metric record per source-target pair. The default re-evaluation JSON retains
the historical target-minus-source delta shape.

## Scientific limits

The legacy fixed-policy transfer harness does not implement environment-agent
coevolution, environment archives, minimal-criterion filtering, or stepping-stone
selection. The standalone PPO-POET module provides a bounded paired population
loop, but its results are not evidence for open-ended coevolution.

Compute scaling fits a least-squares slope of each metric against
`log10(generations)`. It is a descriptive slope, not a power-law exponent.
The ablation driver accepts an explicit seed list or `--seeds 10` and records
that schedule in `ablation_results.json`.

CPU optimisation is measured but not adopted automatically. Run:

```bash
JAX_PLATFORMS=cpu python scripts/benchmark_cpu.py \
  --config configs/scaling_tiny.yaml \
  --compare-paths --benchmark-seeds 10 \
  --output benchmarks/cpu_baseline.json
```

The comparison covers one representative JIT kernel and random-key seed
batching. It does not claim a full-simulator speedup.

## Output shape

`transfer_matrix.json` in re-evaluation mode includes:

```json
{
  "mode": "reeval",
  "metrics": ["action_entropy"],
  "sources": ["source"],
  "targets": ["target"],
  "cells": [
    {"source": "source", "target": "target", "metrics": {"action_entropy": 0.1}}
  ]
}
```

`transfer_matrix.json` in fixed-policy mode includes the same matrix fields,
with `mode` set to `fixed_policy_transfer` and metrics taken directly from the
frozen-policy target evaluation.

`scaling_slopes.json` includes:

```json
{
  "budgets": [10, 20, 40],
  "metrics": {
    "action_entropy": {
      "values": [1.1, 1.2, 1.25],
      "slope_per_log10_gen": 0.21
    }
  }
}
```

Tests exercise fixed-policy shape validation, output-file creation, and
compute-scaling statistics with CPU-cheap synthetic inputs.
