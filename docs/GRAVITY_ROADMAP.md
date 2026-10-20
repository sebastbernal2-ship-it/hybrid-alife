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

Status: **bounded pilot run, negative twice**. Results in
`docs/results/embodied_law_pilot/README.md` and
`docs/results/embodied_law_proprioception/README.md`.

First run: with the flow hidden and only the alignment energy bonus, evolution
changed genomes but did not improve held-out alignment (`+0.001` evolved against
`+0.008` initial). Two plumbing defects were found and fixed first: no births
meant no mutation, and the evaluation measured only the unchanged elite half of
the population.

Second run: proprioception (previous velocity and previous thrust) and a
displacement reward were added. The gate still fails, `0/5` seeds, at reward
`5.0`, reward `50.0`, and 20 generations. Training alignment tracks the sign of
the imposed direction, which is the signature of a fixed action bias, not a
learned response. The reward plumbing is verified; the limiter is evolutionary
search. Selection only mutates dead slots, so the surviving elite freezes.

The claim stays open. The survivor-mutation lever was then tested and also
fails: `0/5` seeds, worst delta `-0.35`. The frozen elite was not the blocker.

The policy-gradient lever was then tested and also fails: `0/5` seeds, worst
delta `-1.90`. The gradient moves the policy (single-direction alignment improves
by more than `0.1`, trained magnitudes reach `0.99`), but it converges to a fixed
action, not a flow-conditioned one. Three protocol variants (warm-up, random
training directions, corrected across-episode advantage baseline) did not change
that. Results in `docs/results/embodied_law_policy_gradient/`.

A gate correction came out of that work: the reversed-law control required
`<= 0.0`, which is backwards. A law-follower re-infers the reversed flow and
stays positive; a fixed action flips sign. The correct condition is
`reversed >= 0.2`. The earlier verdicts are unchanged.

One lever remains: an expressive controller that can represent the
inferred-flow mapping directly. That is the next package.

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

Status: **done and passed**, calibration only. Details in
`docs/results/gravity_study_v1/REPLICATION.md`.

Question: does the result hold on a different machine and a cleared cache?

Outcome: a fresh clone on HiperGator `login7`, with the JAX cache cleared,
reproduced both arms at 5/5 seeds and the same revelations. Per-seed
worst-direction and mean values agree exactly (maximum difference `0.000e+00`).
The artifact is `docs/results/gravity_study_v1/summary_hipergator.json`.

Limits: same JAX version (`0.12.0`) and Python minor (`3.12`) on both machines,
and the same operator. This is a computational replication, not a fully
human-independent one.

### D4. Revelation variants

Status: **done**, calibration only. Results in
`docs/results/revelation_forms/README.md`.

Question: which revelation forms help, and does a false one get corrected?

Outcome: every early true form beats the `+11.00` no-revelation baseline:
exact `+12.00`, demonstration `+12.00`, noisy `+11.94`, symbolic `+11.80`. Late
true revelations add nothing because the estimate is already exact. False
revelations cost about one step early and are then corrected. A message-only
control scores `+1` at best, so the estimate is required.

Channel ordering follows the information carried. Symbolic is lossy because it
names only a cardinal axis. Demonstration is informationally equal to the exact
vector in a uniform world, and the results confirm that.

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
