# Gravity study roadmap

This document owns the prioritized next directions and the accumulated lessons.
`docs/RESEARCH_ALIGNMENT.md` owns the research question and claim boundary.
`docs/results/gravity_study_v1/README.md` owns the v1 findings.

## Ranked directions

### D1. Evolved embodied controller

Question: do the actual evolved embodied organisms exploit an imposed law, not
just a simplified point organism with a linear policy?

Why it matters: this closes the largest gap between the current demonstration
and the original framing of bacteria-like organisms in a world.

Bounded scope: smallest embodied configuration, uniform imposed flow, the same
proprioceptive features the internal-estimate path already uses, held-out
directions.

Controls: no law, reversed law, shuffled flow, and a no-benefit variant.
Avida must stay disabled in causal traces.

Gate: per-direction minimum on the displacement reward, five seeds, with the
same held-out diagonal rule.

Risk: the embodied GRU may not use proprioception, may need observation changes,
and the runner loop is heavier than the point-organism bridge.

Owner files: `src/hybrid_alife/agents/embodied.py` for observation,
`src/hybrid_alife/world/env.py` for the law, `src/hybrid_alife/experiments/runner.py`
for the loop, plus a new `src/hybrid_alife/experiments/latent_law_embodied.py`.

### D2. Reversal adaptation in the simulator

Status: **done**, calibration only. Results in
`docs/results/reversal_adaptation/README.md`.

Question: does the organism adapt when the imposed law reverses mid-episode?

Outcome: the estimate-driven organism re-aligns in exactly one step on every
direction and seed, while a frozen-estimate control never adapts and goes fully
anti-aligned. Regret is `2.00` for the adaptive organism against `16.00` for the
frozen control. A trained estimate policy reproduces this at post-alignment
`+0.99`.

Limitation: the one-step recovery is structural, because the estimate is exact
after one transition. This is fast tracking, not a learned reversal strategy.
The learned-strategy claim belongs to D1.

### D3. Independent replication

Question: does the result hold on a different machine and a cleared cache?

Bounded scope: run `scripts/run_gravity_study.py` on HiperGator with the same
seed schedule, cleared caches, and recorded provenance, then compare to the v1
artifact.

Gate: the same gate outcomes, with machine, backend, and dependency versions
recorded.

Run this last so it replicates the complete state, not an intermediate one.

### D4. Revelation variants

Question: which revelation forms help, and does a false one get corrected?

Bounded scope: demonstration, noisy sensory hint, symbolic message, and delivery
timing. Measure reward and self-correction.

Gate: true variants beat no revelation, and a false revelation is corrected
within one step.

## Execution order

1. D2, because it is fast and extends the existing simulator bridge.
2. D4, because it is fast and extends the existing revelation code.
3. D1, scoped in parallel from the start because it is the largest and riskiest.
4. D3, last, against the complete state.

## Accumulated lessons

These cost real time to discover. Keep them.

- The reward must have an improvement path. A target-position reward produced a
  needle optimum: one isolated good schedule, every neighbour far worse, so no
  incremental learner could find it. A displacement reward fixed this.
- The action space needs an informative mode. Nine discrete thrusts let the
  policy learn a broad mixture whose mean was good and whose mode was arbitrary.
  Continuous thrust with a normalized direction fixed this.
- The learner must be stable on this task. Plain policy gradient collapsed to a
  fixed direction. Reward-weighted regression over the linear mean was reliable.
- Proprioception makes latent-force inference exact: `velocity - friction *
  previous_velocity - acceleration * previous_thrust` recovers the law.
- Never average signed alignment over antipodal directions. It cancels to zero
  and hides per-direction failure. Report per direction, always.
- A mean across directions can hide a catastrophic single-direction failure.
  Make the minimum across directions the primary gate.
- Tests that train models are slow. Keep decisive gates small in the default
  suite and put full-size runs in `scripts/`.
- The CI job has a wall-clock timeout. The suite is about 14 minutes. Raising the
  ceiling to 35 minutes and trimming the two slow simulator tests kept CI green.
- The mutating native worktree subagent path failed repeatedly with exit code
  143 and produced no files. Read-only subagents worked. Prefer read-only scouts
  for design, and implement bounded mutations in-session until that path is
  fixed.

## Vision alignment

The original question was whether simple artificial organisms can figure out an
imposed world condition, or at least unconsciously exploit it, without needing
to name or represent it.

- D1 tests that question on the real evolved organisms rather than a simplified
  proxy, which is the strongest form of the claim.
- D2 tests adaptation, the difference between exploiting a fixed law and
  tracking a changing one.
- D4 tests the revelation idea: whether being told the law helps, and whether a
  misleading message is overridden by the organism's own estimate.
- D3 tests whether the whole result survives an independent machine.

None of these directions supports a consciousness claim. They support behavioural
claims about exploiting and adapting to an imposed regularity.
