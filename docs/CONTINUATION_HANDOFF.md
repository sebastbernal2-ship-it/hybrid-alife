# Continuation handoff: hybrid-alife

Purpose: a new session with zero context can continue this project without
re-deriving state. This file is the checkpoint owner. Deeper detail lives in the
result documents it points to.

## Where things stand

- Repository: https://github.com/sebastbernal2-ship-it/hybrid-alife
- `main` merged state: `2c33449` (PR #26), with this handoff committed on top.
- Working tree clean, tests pass (396 at the checkpoint), CI green on Python 3.11 and 3.12.
- Research question and claim boundary: `docs/RESEARCH_ALIGNMENT.md`
- Ranked directions and accumulated lessons: `docs/GRAVITY_ROADMAP.md`
- Project memory and sharp edges: `AGENTS.md`

## Results on main

| Direction | Outcome | Evidence |
| --- | --- | --- |
| Calibration end to end | 5/5 seeds, worst direction +9.94, mean +10.85 | `docs/results/gravity_study_v1/` |
| Simulator internal estimate | 5/5 seeds, worst direction +10.06, mean +10.92 | same |
| Independent HiperGator replication | passed, exact per-seed agreement | `docs/results/gravity_study_v1/REPLICATION.md` |
| D2 reversal adaptation | one-step recovery, frozen control never adapts | `docs/results/reversal_adaptation/` |
| D4 revelation forms | positive, channel-ordered, false ones self-correct | `docs/results/revelation_forms/` |
| D1 evolved embodied | negative pilot, plumbing fixed, claim open | `docs/results/embodied_law_pilot/` |
| D1 proprioception | negative, 0/5 seeds at every budget tried | `docs/results/embodied_law_proprioception/` |
| D1 survivor mutation | negative, 0/5 seeds, frozen elite not the blocker | `docs/results/embodied_law_survivor_mutation/` |
| D1 policy gradient | negative, 0/5 seeds, gradient reaches a fixed action | `docs/results/embodied_law_policy_gradient/` |
| D1 linear controller | negative, 0/5 seeds, representation sufficient, search is the limiter | `docs/results/embodied_law_linear_policy/` |
| D1 unit-flow normalization | negative, 0/5 seeds, selection scheme is the limiter | `docs/results/embodied_law_unit_flow/` |
| D1 min-alignment fitness | negative, 0/5 seeds, exposes a no-op selection operator | `docs/results/embodied_law_min_alignment/` |
| D1 rank replacement | negative, 0/5 seeds, stuck in an anti-aligned basin | `docs/results/embodied_law_rank_replacement/` |
| D1 random immigrants | negative, 0/5 seeds, sparse-credit problem | `docs/results/embodied_law_immigrants/` |
| D1 flow-only head | **positive pilot, 4/5 seeds**; seed 1 clears at 20 generations | `docs/results/embodied_law_flow_head/` |
| D1 confirmation (20 gen, preregistered) | **4/5 seeds**; threshold not met, failing seed changed | `docs/results/embodied_law_confirmation/` |
| D1 absolute floor (preregistered) | **3/5 seeds**; seeds 1,2 below 0.5, seed 4 passes | `docs/results/embodied_law_absolute_confirmation/` |

The three studies reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py --output docs/results/reversal_adaptation/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py --output docs/results/revelation_forms/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 --generations 8 --population 16 --steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0 --output docs/results/embodied_law_proprioception/summary.json
```

## Current state: D1 4/5 bounded claim accepted

The captain accepted the 4/5 bounded claim on 2026-10-21:

**evolved embodied organisms behaviorally exploit the imposed law in a bounded
pilot, 4 of 5 seeds, confirmed at two budgets, under a four-parameter movement
head.**

The relative-improvement margin is seed-sensitive: the comparison is a random
initial population that can align on a diagonal by chance. Seed 4 missed `0.2` by
`0.016` while its own alignment was `+0.598` and its reversed-law alignment was
`+0.598`.

## Last package: absolute-floor confirmation (3/5, claim unchanged, merged)

The preregistered absolute-floor confirmation ran with the declared budget. Gate
per held-out diagonal: evolved at least `0.5` and reversed at least `0.2`.

Result: **3/5 seeds**.

| Seed | Passed | Minimum evolved | Minimum reversed |
| --- | --- | --- | --- |
| 0 | yes | +0.616 | +0.616 |
| 1 | no | +0.480 | +0.480 |
| 2 | no | +0.392 | +0.392 |
| 3 | yes | +0.539 | +0.539 |
| 4 | yes | +0.598 | +0.598 |

Seeds 1 and 2 are law-following but below the floor. Seed 4 passes the floor and
fails the initial-relative delta. The two criteria are complementary, and neither
reaches `5/5` at this budget. See
`docs/results/embodied_law_absolute_confirmation/README.md`.

The accepted claim is unchanged: 4 of 5 seeds, confirmed at two budgets, under a
four-parameter movement head. The absolute-floor five-seed threshold remains open.
No retune was made.

Both bounded results are accepted and merged:

- 4/5 on the initial-relative delta, confirmed at two budgets.
- 3/5 on the absolute floor.

The two criteria are complementary and neither reaches `5/5` at this budget. The
accepted claim is unchanged.

No work is in flight. The next action needs the captain's word. Options:

1. Move to the next research direction (see `docs/RESEARCH_ALIGNMENT.md` and the
   ranked list below).
2. Preregister a new confirmation run in a new session with a stated change: more
   seeds (for example 10), a larger generation budget, or a different criterion.

Do not relaunch or retune the settled runs silently.

## Constraints and traps learned the hard way

- Run locally with `JAX_PLATFORMS=cpu`. The suite takes about 10 to 15 minutes; CI has a 35-minute job ceiling.
- Keep decisive gate tests small and put full-size training runs in `scripts/`.
- Never report a mean across gravity directions as the primary result. Gate on the minimum across directions.
- Never average signed alignment across antipodal directions; it cancels to zero.
- Avoid reward designs with an isolated optimum. That needle looked like a learner failure and was not.
- Prefer continuous actions over a small discrete set when direction is the target, to avoid a mode-versus-mean failure.
- Two silent plumbing bugs already cost a day in D1: no births meant no mutation, and evaluation resurrected only the elite half of the population. Always verify the control baseline actually differs from the treatment.
- The embodied selection scheme only mutates dead slots. Once one clone dominates, survivors freeze and later generations change nothing. Do not assume a longer run explores more.
- A gate can pass by accident if the metric tracks the imposed direction's sign rather than a learned response. Always include the reversed-law control, and get its direction right: a law-follower re-infers the reversed flow and stays positive, while a fixed action flips sign. The condition is `reversed >= margin`.
- A per-episode advantage baseline erases the cross-episode signal that separates a fixed action from a flow-follower. Baseline across episodes at each timestep, matching `latent_law_continuous.py`.
- Check feature and initial-genome scale before blaming the search. A mapping that is expressible can still be unreachable when the needed weights are three orders of magnitude from the initialization and the mutation scale.
- Verify the selection operator actually acts. A fitness that is never used looks exactly like a bad fitness. If every result is identical across fitness changes, the operator is the suspect.
- Verify a flag reaches the config that actually runs. A flag dropped in an inner config call looks exactly like a flag that had no effect.
- When a dense genome cannot find a sparse solution, shrink the genome before adding more mutation. A four-parameter head answers whether the search can learn a sign at all.
- A positive result is a claim boundary, not a headline. State the seed count, the budget, and the one seed that missed. Do not call a 4/5 pilot a 5/5 result.
- HiperGator access uses a persistent ControlMaster socket at `~/.ssh/hpg-%r@%h:%p`; plain `ssh hipergator` reuses it. If it prompts, reconnect interactively once.

## Open decisions

- The PPO-POET archive claim stays closed as `insufficient_evidence` unless the
  captain reopens it deliberately.
- Any claim about consciousness, sentience, or symbolic representation is out of
  scope for this study and must not be inferred from these results.

## Earlier history

Older campaign handoffs and field-discovery history were superseded by the
results above. The archived artifacts remain under `docs/results/`, including
`poet_campaign_final_v6/` and the field-discovery pilots.
