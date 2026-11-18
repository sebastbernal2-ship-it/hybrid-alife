# Flow-head law study under an LBM-derived law profile

Status: **bounded pilot. 15 seeds: LBM profile 11/15 against uniform baseline 14/15 on the
same code. The gap is not statistically distinguishable (Fisher exact p = 0.33). Not a claim.**

## Question

The flow-head result (`../embodied_law_flow_head/`) imposes a spatially uniform law.
Does it survive a law with realistic structure: a Poiseuille-like speed profile with
no-slip walls and an obstacle that deflects the flow, from the native D2Q9 solver?

## Change

`--law-profile lbm` in `scripts/run_embodied_law_study.py`. The solver field
(`lbm_law_profile`, peak speed 1, obstacle radius `size/6`) has its +x direction
rotated onto each imposed direction. Only the vectors rotate; the wall and obstacle
geometry stays fixed, so the field is a structured law, not a rotated channel.
Everything else is the recorded flow-head protocol (5 seeds, 8 generations, 16 agents,
14x14, margin gate 0.2, reversed-law floor 0.2).

## Results

| Seed | Uniform min_delta / min_reversed | LBM min_delta / min_reversed |
|---|---|---|
| 0 | 0.659 / 0.567 pass | 0.549 / 0.477 pass |
| 1 | 0.193 / 0.411 fail | -0.464 / -0.030 fail |
| 2 | 0.482 / 0.615 pass | 0.403 / 0.489 pass |
| 3 | 0.376 / 0.547 pass | 0.428 / 0.623 pass |
| 4 | 0.210 / 0.576 pass | 0.198 / 0.632 fail |

The uniform column reproduces `embodied_law_flow_head` exactly (4/5, seed 1 at 0.193).

## Reading (first five seeds)

- Seed 4 misses the margin by 0.002 and its reversed-law alignment is the strongest
  of the set, so it is the same margin-sensitivity failure the baseline already shows.
- Seed 1 is different: minimum delta -0.464 and reversed alignment -0.030 mean the
  evolved policy is worse than the random start and does not follow the reversed law.
  That is a fixed-action or anti-aligned outcome, and it appears only under the
  structured law. It is one seed of five.
- With five seeds this cannot separate "structure hurts" from seed variance. The
  limited claim is that the mechanism still works on 3 of 5 seeds under a structured
  law, and that one seed fails in a way the uniform law did not produce.
- Wall rows and obstacle cells carry zero flow, so fewer agents are scored there.

## Extension to 15 seeds (seeds 5-14 added, same protocol)

| Condition | Pass | Failing seeds (min_delta / min_reversed) |
|---|---|---|
| Uniform | 14/15 | 1 (0.193 / 0.411) |
| LBM profile | 11/15 | 1 (-0.464 / -0.030), 4 (0.198 / 0.632), 5 (0.044 / 0.030), 9 (-0.006 / 0.403) |

Medians of min_delta are indistinguishable (0.393 uniform, 0.408 LBM). The LBM
failures are more varied: one anti-aligned seed (1), one near-miss (4), and two where
evolution did not improve on the random start (5, 9). Fisher exact on 14/1 against
11/4 gives p = 0.33, so 15 seeds do not separate a real cost of structure from seed
variance. Per-seed rows: `seeds0-14_uniform.json`, `seeds0-14_lbm.json`.

## Next

A variant where
the geometry rotates with the law instead of only the vectors.
