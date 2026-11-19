# Flow-head law study under an LBM-derived law profile

Status: **bounded pilot, 15 seeds per condition. Pass counts: uniform 14/15, LBM
vectors-only 11/15, rotated geometry 10/15, plain channel 10/15, rotated geometry with
low-flow cells unscored 7/15. Pass counts alone do not separate the conditions (Fisher exact,
uniform against each of the first three: p = 0.17 to 0.33; the flow-floor condition, p =
0.014). A paired test on the continuous min_delta is more powerful and does resolve a lower
min_delta than uniform for rotated geometry, plain channel and the flow-floor condition, but
not for vectors-only (see "Paired test on min_delta"). Not a claim.**

## Question

The flow-head result (`../embodied_law_flow_head/`) imposes a spatially uniform law.
Does it survive a law with realistic structure: a Poiseuille-like speed profile with
no-slip walls and an obstacle that deflects the flow, from the native D2Q9 solver?

## Change

`--law-profile lbm` in `scripts/run_embodied_law_study.py`. The solver field
(`lbm_law_profile`, peak speed 1, obstacle radius `size/6`) has its +x direction
rotated onto each imposed direction. Only the vectors rotate; the wall and obstacle
geometry stays fixed, so the field is a structured law, not a rotated channel.
Everything else is the recorded flow-head protocol (8 generations, 16 agents, 8 steps per
generation, 14x14, margin gate 0.2, reversed-law floor 0.2). Seeds 0-4 were run first and
extended to seeds 0-14 for every condition below.

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
vectors-only (seed 1). Three are near-misses within 0.031 of the margin (seeds 1, 7, 9: 0.027, 0.031, 0.013), and seed 13
clears the margin but misses the reversed floor by 0.019. Per-seed rows:
`seeds0-14_lbm_geometry.json`.

Reading: the structure of the law costs pass rate in both variants (14/15 to 11/15 and
10/15), but the pass-count difference is not resolved at 15 seeds; the paired min_delta test
in the next section resolves it for the rotated geometry and not for vectors-only. The
rotated geometry removes the fixed-geometry explanation for the vectors-only failures,
since it fails at least as often. The failures are margin-type, not collapse-type.
The gates were set on uniform-law behaviour and the structured laws have zero-flow cells
that reduce scored agents, which may account for part of the lower margin.

## Paired test on min_delta

Added after an audit that recomputed every number in this file from the per-seed JSONs.
Pass counts throw away most of the information in `min_delta`, and the seeds are paired (a
seed fixes the initial population), so a paired Wilcoxon on `min_delta` against the uniform
law is the more powerful test. Four tests, Holm-corrected:

| Condition vs uniform | Mean change in min_delta | Seeds worse / better | Wilcoxon p | Holm p (4) |
|---|---|---|---|---|
| LBM vectors-only | -0.109 | 10 / 5 | 0.208 | 0.208 |
| LBM rotated geometry | -0.115 | 14 / 1 | 0.0003 | 0.0012 |
| LBM plain channel (no obstacle) | -0.130 | 12 / 3 | 0.0043 | 0.0085 |
| Rotated geometry, flow floor 0.3 | -0.218 | 13 / 2 | 0.0012 | 0.0035 |

So the earlier "individually unresolved" is true of pass counts only. A lower min_delta than
uniform is resolved for rotated geometry, plain channel and flow-floor, and not for
vectors-only. Plain channel against rotated geometry: -0.015 (p = 1.0), so the obstacle adds
nothing resolvable. The flow-floor condition against rotated geometry is -0.103 on 11 of 15
seeds (p = 0.030, uncorrected; Holm over three such comparisons 0.09). Caveat: `min_delta`
is evolved minus the random initial population scored under the same law, so a structured
law can lower it by changing either term; no analysis here separates the two (the live-fluid
mechanism study, `exp/lbm-live-mechanism`, found that for the monopole-coupled fluid the loss
came from the baseline side). Reproduce with `scripts/audit_lbm_results.py`
(on `exp/lbm-live-sweep-rescored`).

## Confound checks (15 seeds each, rotated geometry)

Two follow-ups targeted the zero-flow-cell explanation and the obstacle.

| Condition | Pass | Median min_delta | Failing seeds (min_delta / min_reversed) |
|---|---|---|---|
| Plain channel, no obstacle (`--no-obstacle`) | 10/15 | 0.308 | 1 (0.099 / 0.267), 4 (0.015 / 0.421), 7 (0.124 / 0.414), 10 (-0.151 / -0.273), 14 (0.090 / 0.195) |
| Obstacle, cells below 0.3 x magnitude unscored (`--flow-floor 0.3`) | 7/15 | 0.145 | 1 (0.145 / 0.293), 3 (0.142 / 0.319), 5 (0.096 / -0.001), 6 (-0.047 / 0.122), 7 (-0.219 / 0.275), 8 (0.094 / 0.369), 9 (0.035 / 0.216), 10 (0.134 / 0.317) |

- **The obstacle is not the cause.** A plain channel passes 10/15, the same as the obstacle
  (10/15). The speed profile across the channel is enough to lower the pass rate.
- **The zero-flow-cell explanation is not supported.** Excluding low-flow cells from
  scoring (and from the min-alignment fitness, which shares the validity mask) made the
  pass rate worse, 7/15, not better (against rotated geometry's 10/15 the pass-count difference
  is not resolved, Fisher p = 0.46; the paired min_delta difference is in the section above).
  Uniform against flow-floor: Fisher p = 0.014, the only pairwise pass-count difference here
  below 0.05. Caveat: the floor also changes training fitness and
  reduces the agents scored per direction, so it conflates two effects and is noisier. It
  does not isolate the scoring effect on its own.
- Plain channel against uniform is p = 0.17 on pass counts, but the paired min_delta test gives
  p = 0.0043 (Holm 0.0085), so for that condition a lower min_delta is resolved.

Taken together: structured laws pass at 10-11 of 15 across three variants against 14 of 15
uniform, consistently lower on pass counts (individually unresolved) and, on the paired
min_delta test, resolved lower for rotated geometry and plain channel. A candidate
mechanism, untested for these laws: the spatially varying speed makes the proprioceptive
estimate `(velocity - thrust)` change as agents move between cells, so a four-parameter
head tuned on constant flow faces a noisier signal. (For the live-coupled fluids, where it
was measured, estimate error did not explain the loss: `exp/lbm-live-estimate`.) An equally
open alternative from the live-fluid study is that the baseline term of `min_delta` changes
under a structured law.

## Next

Measure the unit-flow estimate's error against the true local direction for these structured
laws (it has been measured only for live-coupled fluids), score the initial and evolved
populations separately to see which side of `min_delta` moves, and run about 40 seeds per arm
to resolve the pass-count gaps.

## Files and reproduction

Per-seed rows (list of dicts, one per seed): `seeds0-14_uniform.json`, `seeds0-14_lbm.json`
(vectors-only), `seeds0-14_lbm_geometry.json`, `seeds0-14_lbm_channel_geometry.json`,
`seeds0-14_lbm_geometry_flow_floor.json`. `summary_uniform.json` and `summary_lbm.json` are
the first-pass 5-seed summaries (seeds 0-4) and are superseded by the files above. Each
condition was run with the recorded flow-head flags (`--generations 8 --population 16
--steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0
--unit-flow-features --flow-head --fitness-mode min_alignment --fitness-steps 6
--rank-replacement --immigrant-fraction 0.25`) plus `--law-profile uniform | lbm |
lbm-geometry`, with `--no-obstacle` for the plain channel and `--flow-floor 0.3` for the
flow-floor condition. The uniform column reproduces `embodied_law_flow_head` on seeds 0-4 to
5e-8 (checked by the audit script). Nothing here is unreproducible from the committed files.
