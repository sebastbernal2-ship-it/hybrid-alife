# Flow-head law study under an LBM-derived law profile

Status: **bounded pilot, 15 seeds per condition: uniform 14/15, LBM vectors-only 11/15, LBM
rotated-geometry 10/15. No pairwise gap is statistically distinguishable (smallest Fisher
exact p = 0.17, uniform against rotated-geometry). Not a claim.**

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

## Rotated-geometry variant (15 seeds)

`--law-profile lbm-geometry` rotates the whole channel about the domain centre (bilinear
sampling, periodic along the channel, zero beyond the walls), so walls and obstacle move
with the law. Cardinal directions are exact quarter turns (`tests/test_law_profile.py`).

| Condition | Pass | Failing seeds (min_delta / min_reversed) |
|---|---|---|
| LBM rotated geometry | 10/15 | 1 (0.173 / 0.337), 7 (0.169 / 0.539), 9 (0.187 / 0.508), 10 (0.086 / 0.245), 13 (0.381 / 0.181) |

Median min_delta is 0.316 (uniform 0.393, vectors-only 0.408). Fisher exact: against
uniform p = 0.17, against vectors-only p = 1.0. Every rotated-geometry failure still has
positive reversed-law alignment, so none is the anti-aligned outcome seen once under
vectors-only (seed 1). Three are near-misses within 0.03 of the margin, and seed 13
clears the margin but misses the reversed floor by 0.019. Per-seed rows:
`seeds0-14_lbm_geometry.json`.

Reading: the structure of the law costs pass rate in both variants (14/15 to 11/15 and
10/15) but the direction is consistent and the size is not resolved at 15 seeds. The
rotated geometry removes the fixed-geometry explanation for the vectors-only failures,
since it fails at least as often. The failures are margin-type, not collapse-type.
The gates were set on uniform-law behaviour and the structured laws have zero-flow cells
that reduce scored agents, which may account for part of the lower margin.

## Confound checks (15 seeds each, rotated geometry)

Two follow-ups targeted the zero-flow-cell explanation and the obstacle.

| Condition | Pass | Median min_delta | Failing seeds (min_delta / min_reversed) |
|---|---|---|---|
| Plain channel, no obstacle (`--no-obstacle`) | 10/15 | 0.308 | 1 (0.099 / 0.267), 4 (0.015 / 0.421), 7 (0.124 / 0.414), 10 (-0.151 / -0.273), 14 (0.090 / 0.195) |
| Obstacle, cells below 0.3 x magnitude unscored (`--flow-floor 0.3`) | 7/15 | 0.145 | 1, 3, 5 (0.096 / -0.001), 6, 7 (-0.219 / 0.275), 8, 9, 10 |

- **The obstacle is not the cause.** A plain channel passes 10/15, the same as the obstacle
  (10/15). The speed profile across the channel is enough to lower the pass rate.
- **The zero-flow-cell explanation is not supported.** Excluding low-flow cells from
  scoring (and from the min-alignment fitness, which shares the validity mask) made the
  pass rate worse, 7/15, not better. Uniform against flow-floor: Fisher p = 0.014, the only
  pairwise difference here below 0.05. Caveat: the floor also changes training fitness and
  reduces the agents scored per direction, so it conflates two effects and is noisier. It
  does not isolate the scoring effect on its own.
- Plain channel against uniform is p = 0.17, so the cost of structure is still not
  resolved at 15 seeds for that condition alone.

Taken together: structured laws pass at 10-11 of 15 across three variants against 14 of 15
uniform, consistently lower but individually unresolved. A candidate mechanism, untested
here: the spatially varying speed makes the proprioceptive estimate `(velocity - thrust)`
change as agents move between cells, so a four-parameter head tuned on constant flow
faces a noisier signal. Testing it needs the estimate's variance per condition.

## Next

Measure the unit-flow estimate's error against the true local direction per condition,
and run about 40 seeds per arm to resolve the plain-channel gap.
