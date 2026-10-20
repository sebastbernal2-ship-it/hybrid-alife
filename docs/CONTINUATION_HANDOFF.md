# Continuation handoff: hybrid-alife

Purpose: a new session with zero context can continue this project without
re-deriving state. This file is the checkpoint owner. Deeper detail lives in the
result documents it points to.

## Where things stand

- Repository: https://github.com/sebastbernal2-ship-it/hybrid-alife
- `main` merged state: `52a0839` (PR #19), with this handoff committed on top.
- Working tree clean, tests pass (377 at the checkpoint), CI green on Python 3.11 and 3.12.
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

The three studies reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py --output docs/results/reversal_adaptation/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py --output docs/results/revelation_forms/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 --generations 8 --population 16 --steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0 --output docs/results/embodied_law_proprioception/summary.json
```

## Next bounded package: D1 normalized flow features

All three search levers are ruled out. The linear controller can express the
mapping (hand-set weights align above `0.99`), but evolution cannot reach the
needed weight scale: the flow features are about `0.01`, the needed linear
weights are about `100`, genomes initialize at `0.1`, and mutation has standard
deviation `0.05` over 8 generations. See
`docs/results/embodied_law_linear_policy/README.md`.

Goal: remove the scale gap by giving the controller a scale-free flow feature,
then rerun the same gate.

Steps:

1. Add `EmbodiedConfig.unit_flow_features` (default off). When on, append two
   features: `(previous_velocity - previous_thrust) / max(norm, eps)`, the unit
   flow estimate. This is order one and scale-free, and it keeps the default
   observation unchanged.
2. Test that the feature is unit length after a step, scale-invariant across two
   flow magnitudes, and that a linear controller with order-one weights aligns
   above `0.99`.
3. Run the evolution pilot with the linear controller and the unit-flow feature
   on the four cardinals, evaluated frozen on the held-out diagonals with the
   initial-policy and corrected reversed-law controls.
4. Gate: per-direction held-out alignment beats the initial population by `0.2`,
   and the reversed-law alignment is at least `0.2`. State the budget first.
5. Also run one seed at 20 generations as a diagnostic, reported separately, not
   silently folded into the main budget.

Stop condition: if evolution still fails with order-one features, the limiter is
the selection scheme itself, and the next design must change how fitness is
assigned or how genomes are explored, not the observation.

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
