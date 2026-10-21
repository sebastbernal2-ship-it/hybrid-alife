# Absolute-floor confirmation: results

Status: **3/5 seeds. The absolute-floor threshold was not met, so the claim is not
upgraded.**

The budget and gate below were declared before launch and were not changed.

## Declared criterion

Per held-out diagonal:

- evolved alignment at least `0.5`, and
- reversed-law alignment at least `0.2`.

The initial-relative delta stays reported per seed as a diagnostic, not as the gate.

Declared budget: 5 seeds, 20 generations, 16 agents, 8 steps per generation, 14x14
world, proprioception on, unit-flow feature, flow head, min-alignment fitness with 6
frozen steps per direction, rank replacement with `elite_fraction = 0.5`, immigrant
fraction `0.25`, displacement reward `5.0`, eval steps `12`.

## Results

| Seed | Passed | Minimum evolved | Minimum reversed-law | Minimum delta |
| --- | --- | --- | --- | --- |
| 0 | yes | +0.616 | +0.616 | +0.657 |
| 1 | **no** | +0.480 | +0.480 | +0.283 |
| 2 | **no** | +0.392 | +0.392 | +0.258 |
| 3 | yes | +0.539 | +0.539 | +0.368 |
| 4 | yes | +0.598 | +0.598 | +0.184 |

**3/5 seeds pass.**

## What the failures are

Seeds 1 and 2 are law-following, not fixed actions:

- Every evolved alignment is positive on all four diagonals.
- Reversed-law alignment equals the evolved alignment, so the policy re-infers the
  reversed flow.
- Seeds 1 and 2 simply do not reach `0.5` on their worst diagonal: `0.480` and
  `0.392`.

So the failures are magnitude, not sign.

## The two criteria are complementary

| Criterion | Seeds passing | Seed that fails | Why |
| --- | --- | --- | --- |
| Initial-relative delta >= 0.2 | 4/5 | 4 (0.184) | random initial was strong on two diagonals |
| Absolute floor >= 0.5 | 3/5 | 1 (0.480), 2 (0.392) | evolved policy is positive but below 0.5 on one diagonal |

Seed 4 passes the absolute floor and fails the delta. Seeds 1 and 2 pass the delta
and fail the absolute floor. No seed set passes both at 5/5 with this budget.

## Recording rule followed

Every seed is reported. No retune. No budget change.

## Claim

The accepted claim is unchanged: evolved embodied organisms behaviorally exploit the
imposed law in a bounded pilot, **4 of 5 seeds, confirmed at two budgets**, under a
four-parameter movement head.

The absolute-floor five-seed threshold remains open.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 20 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --unit-flow-features --flow-head \
  --fitness-mode min_alignment --rank-replacement --immigrant-fraction 0.25 \
  --min-evolved 0.5 \
  --output docs/results/embodied_law_absolute_confirmation/summary.json
```
