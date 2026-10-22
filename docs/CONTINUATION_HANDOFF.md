# Continuation handoff: hybrid-alife

Purpose: a new session with zero context can continue this project without
re-deriving state. This file is the checkpoint owner. Deeper detail lives in the
result documents it points to.

## Where things stand

- Repository: https://github.com/sebastbernal2-ship-it/hybrid-alife
- `main` merged state: `d384c37` (PR #36), with this handoff committed on top.
- Working tree clean, tests pass (420 at the checkpoint), CI green on Python 3.11 and 3.12.
- Research question and claim boundary: `docs/RESEARCH_ALIGNMENT.md`
- Ranked directions and accumulated lessons: `docs/GRAVITY_ROADMAP.md`
- Project memory and sharp edges: `AGENTS.md`

## Current direction: Bibites-like vivarium

The captain's real target is a Bibites-like living world first, with the research
question embedded inside it later. The D1 work is the completed research core, not
the product.

- `src/hybrid_alife/vivarium.py` owns seasons, periodic fertile food patches,
  corpses that decay into food, births and deaths, lineage colours, trails, and a
  Pillow renderer plus ffmpeg video writer.
- `scripts/run_vivarium.py` runs the world.
- `docs/results/vivarium_v0/` holds the 30 second video and poster.
- Tests: `tests/test_vivarium.py` (seasons, corpse decay, death leaves a corpse,
  renderer writes a frame).

Decisions:

- **The world stays toroidal.** No walls. The captain chose the torus. The seam is
  cosmetic; creatures wrap left to right and top to bottom. The edge fix made the
  food pattern periodic, keyed trails by slot and lineage, and wrapped the bodies.
- **Bodies and morphology is done** (PR #32). Body genes `body_radius` and
  `body_aspect` inherit and mutate. Larger bodies eat more and pay more cost.
  Oriented ellipse bodies with eyes. Morphology is opt-in so D1 stays exact.
  `docs/results/vivarium_bodies/` holds the video.
- **Live viewer is done** (PR #33). `scripts/serve_vivarium.py` streams MJPEG from
  the Pillow renderer. Open http://127.0.0.1:8765.
- **World-embedded law experiment is done: 1/3 seeds, a fixed bias, not
  law-following.** `docs/results/vivarium_law/` holds the plot and summary. The
  population drifts to -x even with no law. The -x law looks positive because it
  reinforces that bias; the +x law does not turn the population. This matches the
  D1 history: the 5/5 claim needed the flow head, min-alignment fitness, and rank
  replacement, which the natural world does not use.
- **Comparison is done: both arms 1/3, both a fixed bias.** The D1 arm used the
  linear controller because the pure flow head cannot eat and starved. The linear
  arm plus min-alignment fitness and rank replacement still settles into a fixed
  direction bias. Even the passing D1 seed 0 ends at `-0.294` against its +x law:
  it passes the relative gate but is not positively aligned. Results in
  `docs/results/vivarium_law_d1/`.
- **Flow head plus eat gate was built and tested, and it is not viable as a
  survival comparison.** The law conditions survive but both show the same fixed
  bias; the law-absent condition collapses because the movement-only brain cannot
  seek food. No valid absent baseline, so no completed comparison. See
  `docs/results/vivarium_law_flow/`.
- **No work is in flight.** The captain merged the design finding but has not yet
  chosen between:
  1. A hybrid movement head: flow-following plus a simple food gradient, so the
     organism survives in both law and absent conditions.
  2. Accept the negative: no vivarium arm shows robust law-following, and the
     controlled D1 5/5 remains a special setup. Keep the living world as the
     product and the D1 result as the controlled research core.

No work is in flight. The D1 claim remains: 5 of 5 seeds, bounded pilot, confirmed
at two budgets, four-parameter movement head.

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
| D1 ablation | **mechanism identified**; flow head, min-alignment, rank replacement necessary | `docs/results/d1_ablation/` |
| D1 no-immigrant confirmation | **5/5 seeds, claim upgraded**; delta and absolute floor both 5/5 | `docs/results/embodied_law_no_immigrant_confirmation/` |

The three studies reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py --output docs/results/reversal_adaptation/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py --output docs/results/revelation_forms/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 --generations 8 --population 16 --steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0 --output docs/results/embodied_law_proprioception/summary.json
```

## Last package: no-immigrant confirmation (5/5, claim upgraded)

The preregistered no-immigrant confirmation ran with the declared 20-generation,
5-seed budget. Result: **5/5 seeds pass** the initial-relative gate, and **5/5**
pass the secondary absolute floor.

| Seed | Passed | Evolved minus initial | Evolved (worst) | Minimum reversed |
| --- | --- | --- | --- | --- |
| 0 | yes | +0.734 | +0.694 | +0.694 |
| 1 | yes | +0.513 | +0.551 | +0.551 |
| 2 | yes | +0.697 | +0.830 | +0.830 |
| 3 | yes | +0.516 | +0.687 | +0.687 |
| 4 | yes | +0.377 | +0.792 | +0.792 |

Training alignment rises to about `0.95` and stays, and reversed-law alignment is
positive in every seed and equals the evolved alignment. See
`docs/results/embodied_law_no_immigrant_confirmation/README.md`.

## Claim, upgraded

Evolved embodied organisms behaviorally exploit the imposed law: **5 of 5 seeds, 20
generations, under a four-parameter movement head with rank replacement and no
immigrants. The initial-relative gate and the absolute floor both pass 5/5.**

Boundary: one uniform-flow environment, held-out diagonals, five seeds, one machine.
No consciousness or symbolic-representation claim.

## Current state

No work is in flight. Open follow-ups, on the captain's word:

1. Second-machine replication of the 5/5 result.
2. Open a new research question from `docs/RESEARCH_ALIGNMENT.md` with a fresh spec.
3. Stop here.

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
