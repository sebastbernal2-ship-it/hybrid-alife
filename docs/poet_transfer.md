# POET / Transfer / Compute-Scaling — v1 Semantics

This document describes the v1 transfer and compute-scaling harness for
hybrid-alife. It is deliberately minimal so it can run on a smoke compute
budget (~tens of seconds per cell on CPU) while still producing real,
machine-checkable numbers.

## What this harness does

There are two top-level scripts:

- `scripts/run_transfer_matrix.py` — checkpoint-level transfer:
  - For each `--source-config`, run a short training/evaluation (or load an
    existing run dir if `--source-runs` is provided).
  - For each `--target-config`, re-evaluate the same final metrics under the
    target world settings as a checkpoint-level transfer.
  - Writes a JSON matrix (`transfer_matrix.json`) and a markdown table
    (`transfer_matrix.md`) of metric deltas (target − source).

- `scripts/run_compute_scaling.py` — compute-scaling slope:
  - For each `--budget` (generation count), runs the same fixed-world config.
  - Fits a least-squares slope of `metric vs log10(generations)` for each
    metric in `--metrics` and writes `scaling_slopes.json` plus a markdown
    summary.

The placeholder/seed configs used by these scripts are:

- `configs/transfer_source.yaml` — small source world.
- `configs/transfer_target_uniform.yaml` — same dynamics, no resource flow
  (uniform field) as the transfer target.
- `configs/scaling_tiny.yaml` — fixed-world, tiny per-cell compute used to
  sweep generation budgets.

## v1 limitations (read this before citing results)

This is **not** a full POET implementation. In particular:

1. **No co-evolution of environments and agents.** Environments are static
   configs picked by the user. Real POET dynamically generates and selects
   environments based on agent progress.
2. **Checkpoint-level transfer, not policy transfer.** A "transferred"
   agent is the final metric snapshot evaluated under target world settings
   from the same run — we do not yet re-roll the trained genomes into a new
   world rollout. This means the transfer matrix measures *config robustness
   of the metric* rather than *policy generalisation*.
3. **Compute scaling uses a 1D log-fit.** The slope is informative but is
   not a power-law exponent. Confidence intervals are not computed at v1.
4. **No minimal-criterion filtering or stepping-stone bookkeeping.**

These limitations are intentional. The harness is designed so the JSON and
markdown outputs have a stable shape; later versions can fill in real policy
transfer and POET dynamics without breaking the I/O.

## Output shape

`transfer_matrix.json`:

```json
{
  "metrics": ["action_entropy", "mean_avida_merit", "qd_score", ...],
  "sources": ["transfer_source"],
  "targets": ["transfer_source", "transfer_target_uniform"],
  "cells": [
    {"source": "transfer_source", "target": "transfer_source",
     "metrics": {"action_entropy": 1.23, ...}}
  ]
}
```

`scaling_slopes.json`:

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

Tests in `tests/test_transfer_scaling.py` exercise the output-file creation
and shape using synthetic metrics so the suite stays CPU-cheap.
