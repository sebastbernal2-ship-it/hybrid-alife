# Reversed-law audit and null-selection control

## Finding: an identity by design, not an independent control

`evaluate_frozen_alignment(reverse_law=True)` only negates the imposed direction. Every
evaluation calls `initialize_sim` with the same config and seed, copies the same genomes in,
and restarts any live fluid. The study's diagonals are closed under negation, so
`reversed[d]` is exactly the rollout `evolved[-d]`. It holds for every law profile (uniform,
LBM, rotated geometry, live fluid), whether or not the two fields are mirror images.

`python scripts/audit_reversed_law.py`: 373 stored rows (368 earlier ones plus the 5 null
rows below, including diagnostics and files that overlap by seed), 1492 values, **0 differ** from the antipodal evolved value,
maximum difference 0.0. `tests/test_reversed_law_control.py` checks the identity on a live
rollout and records the independence claim as a strict expected failure.

## What the clause does

- It carries no new measurement. It is the floor `evolved >= 0.2` on all four diagonals.
- Absolute-floor gate (`evolved >= 0.5`): fully redundant.
- Initial-relative gate: not redundant as a threshold. A hand-set fixed action (`w_flow = 0`,
  bias along x) scores about +0.71 / -0.71 on the diagonals. Against a hand-set anti-aligned
  baseline it beats the 0.2 margin on all four. The floor rejects it, because a fixed action
  cannot be positive on both members of an antipodal pair under the uniform law (also
  checked for the two static LBM laws). So a fixed action does **not** pass the existing gate.

## Verdicts without the clause (stored per-seed data, numbers unchanged)

| Study | With clause | Primary criterion only | Seed that flips (min_delta / min_evolved) |
| --- | --- | --- | --- |
| D1 ablation `no_rank` | 0/5 | 1/5 | 0 (0.364 / 0.136) |
| LBM rotated geometry | 10/15 | 11/15 | 13 (0.381 / 0.181) |
| Every other `embodied_law_*` and ablation file | as stored | unchanged | none |

- The 4/5, 3/5 (absolute) and 5/5 results are unchanged. All negative pilots stay at 0 passes,
  including those recorded under the older `reversed <= 0.0` gate.
- Ablation: `no_rank` still falls short of `full` (1/5 against 4/5), but "drops to 0/5" depends
  on the floor.
- LBM pass counts, Fisher exact without the clause: uniform against rotated geometry p = 0.33
  (stored 0.17); rotated geometry against flow floor p = 0.26 (stored 0.46); rotated geometry
  against vectors-only p = 1.0. No conclusion about significance changes.
- Paired tests on `min_delta` do not use the clause. Statements about `min_reversed` (live and
  drag-sweep READMEs) are statements about minimum evolved alignment.

Replacing the clause with an explicit `evolved >= 0.2` floor reproduces every verdict of the
`reversed >= 0.2` gate. The gate code is left as it was.

## Why a flipped or rotated sensor is not used

The flow head is memoryless and its only input is the flow estimate. Under a uniform law,
rotating that estimate by a multiple of 90 degrees probes the same four inputs the diagonals
already probe, so its score is predictable from the diagonal responses. It would be a useful
intervention for a dense controller with other inputs.

## Independent control: null selection

It tests the claim the gate cannot: that selection on alignment produced the behaviour. It
uses separate training runs. `--fitness-mode null_min_alignment` computes the same
min-alignment scores, then randomly permutes them across slots before selection. The
permutation uses its own generator, so the initial population is matched seed for seed.

Criterion, chosen before launch (not committed beforehand, so not a formal
preregistration): the real run's minimum evolved
alignment exceeds the null's on all 5 seeds (one-sided sign test p = 1/32), and the null
passes the unchanged gate on at most 1 of 5.

Configuration: the 5/5 no-immigrant confirmation (20 generations). The real arm was re-run
and matches the stored `evolved` and `initial` values to 1.6e-7.

| Seed | Real min evolved / min_delta | Null min evolved / min_delta |
| --- | --- | --- |
| 0 | +0.694 / +0.734 | -0.734 / -0.611 |
| 1 | +0.551 / +0.513 | -0.786 / -0.666 |
| 2 | +0.830 / +0.697 | -0.654 / -0.788 |
| 3 | +0.687 / +0.516 | -0.145 / -0.086 |
| 4 | +0.792 / +0.377 | -0.518 / -0.671 |

**The control passes:** real exceeds null on 5/5 seeds; the null passes 0/5, with or without
the reversed clause.

Limits: one configuration, five seeds, one machine, one permutation stream. Energy-gated
births during training remain in the null arm, so some energy-based selection survives
there; the control removes only the end-of-generation use of the scores. It says nothing
about the other studies, and it is not part of the recorded gate.

Reproduce (per-seed rows in `null.json`; use `--fitness-mode min_alignment` for the real arm):

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 \
  --generations 20 --population 16 --steps-per-generation 8 --size 14 --eval-steps 12 \
  --fitness-steps 6 --displacement-reward 5.0 --unit-flow-features --flow-head \
  --rank-replacement --immigrant-fraction 0.0 --fitness-mode null_min_alignment \
  --output docs/results/embodied_law_null_selection/null.json
```
