# Continuation handoff: hybrid-alife

Purpose: a new session with zero context can continue this project without
re-deriving state. This file is the checkpoint owner. Deeper detail lives in the
result documents it points to.

## Where things stand

- Repository: https://github.com/sebastbernal2-ship-it/hybrid-alife
- `main` merged state: `6398357` (PR #15), with this handoff committed on top.
- Working tree clean, tests pass (366 at the checkpoint), CI green on Python 3.11 and 3.12.
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

The three studies reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py --output docs/results/reversal_adaptation/summary.json
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py --output docs/results/revelation_forms/summary.json
```

## Next bounded package: D1 proprioception

Goal: test whether the **actual evolved embodied organisms** exploit an imposed
law, with the inference made possible from their own sensors.

Why: D1's bounded pilot failed because the embodied observation has no
previous-velocity or previous-thrust features, so a hidden law cannot be inferred
from the organism's own motion, and the alignment energy bonus is too weak to
drive selection. See `docs/results/embodied_law_pilot/README.md`.

Steps:

1. Add explicit proprioceptive features to the embodied observation:
   previous velocity and previous thrust, appended in `observe_embodied`, with
   `embodied_observation_dim` updated. This changes controller input and genome
   shapes, so old checkpoints are invalidated. Test the new shape explicitly.
2. Add a displacement-style reward, default off, that adds
   `dot(unit_thrust, unit_law)` per step to energy, so selection is driven by the
   behavior rather than a small bonus. Keep existing dynamics unchanged by default.
3. Re-run the bounded pilot in `src/hybrid_alife/experiments/latent_law_embodied.py`
   with the same held-out diagonals, the initial-population control, and the
   reversed-law control.
4. Gate: per-direction alignment whose minimum beats the initial-population
   control on held-out diagonals. Report per direction, never an antipodal mean.
5. Keep Avida disabled. Keep the flow hidden in the primary condition and use a
   visible-flow variant only as a cue-availability diagnostic.
6. Keep the default pytest suite fast: the pilot runs belong in a script, not in
   more slow tests.

Stop condition: if the new observation shape breaks existing controller tests, or
the pilot cannot beat the initial-population control within the declared budget,
report the negative result and do not enlarge the budget without revising the brief.

## Constraints and traps learned the hard way

- Run locally with `JAX_PLATFORMS=cpu`. The suite takes about 10 to 15 minutes; CI has a 35-minute job ceiling.
- Keep decisive gate tests small and put full-size training runs in `scripts/`.
- Never report a mean across gravity directions as the primary result. Gate on the minimum across directions.
- Never average signed alignment across antipodal directions; it cancels to zero.
- Avoid reward designs with an isolated optimum. That needle looked like a learner failure and was not.
- Prefer continuous actions over a small discrete set when direction is the target, to avoid a mode-versus-mean failure.
- Two silent plumbing bugs already cost a day in D1: no births meant no mutation, and evaluation resurrected only the elite half of the population. Always verify the control baseline actually differs from the treatment.
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
