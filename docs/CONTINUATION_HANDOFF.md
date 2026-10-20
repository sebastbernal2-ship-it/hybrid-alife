# Continuation handoff: hybrid-alife

Purpose: a new session with zero context can continue this project without
re-deriving state. This file is the checkpoint owner. Deeper detail lives in the
result documents it points to.

## Where things stand

- Repository: https://github.com/sebastbernal2-ship-it/hybrid-alife
- `main` merged state: `b53c698` (PR #17), with this handoff committed on top.
- Working tree clean, tests pass (372 at the checkpoint), CI green on Python 3.11 and 3.12.
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

The three studies reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py --output docs/results/reversal_adaptation/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py --output docs/results/revelation_forms/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 --generations 8 --population 16 --steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0 --output docs/results/embodied_law_proprioception/summary.json
```

## Next bounded package: D1 policy-gradient update (lever 1)

The captain picked lever 1 after survivor mutation was ruled out. The proprioception
reward plumbing is verified, and both random-mutation levers fail: `0/5` seeds,
training alignment tracks the direction sign, which is a fixed action bias.

Goal: test whether a gradient update on the embodied controller can move the policy
where evolution did not.

Steps:

1. Reuse the point-organism pattern in
   `src/hybrid_alife/experiments/latent_law_controller.py` for the update shape.
2. Expose the action log-probability from the embodied controller, or add a
   parallel rollout that returns actions, log-probs, and rewards.
3. Write `scripts/run_embodied_policy_gradient.py`: roll out episodes on the
   training cardinals, compute a REINFORCE-style update with a return baseline,
   apply it to the genome pytree, and evaluate frozen on the held-out diagonals.
4. Keep the same gate: per-direction held-out alignment beats the
   initial-population control by `0.2`, and the reversed-law alignment is at most
   `0.0`. State the budget before the run.
5. Report a negative result rather than enlarging the budget silently.

Stop condition: if the gradient cannot beat the initial-population control within
the declared budget, report the negative result and move to the expressive
controller (lever 3) rather than retuning endlessly.

## Constraints and traps learned the hard way

- Run locally with `JAX_PLATFORMS=cpu`. The suite takes about 10 to 15 minutes; CI has a 35-minute job ceiling.
- Keep decisive gate tests small and put full-size training runs in `scripts/`.
- Never report a mean across gravity directions as the primary result. Gate on the minimum across directions.
- Never average signed alignment across antipodal directions; it cancels to zero.
- Avoid reward designs with an isolated optimum. That needle looked like a learner failure and was not.
- Prefer continuous actions over a small discrete set when direction is the target, to avoid a mode-versus-mean failure.
- Two silent plumbing bugs already cost a day in D1: no births meant no mutation, and evaluation resurrected only the elite half of the population. Always verify the control baseline actually differs from the treatment.
- The embodied selection scheme only mutates dead slots. Once one clone dominates, survivors freeze and later generations change nothing. Do not assume a longer run explores more.
- A gate can pass by accident if the metric tracks the imposed direction's sign rather than a learned response. Always include the reversed-law control.
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
