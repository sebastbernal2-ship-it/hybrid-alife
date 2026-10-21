# Confirmation run: results

Status: **4/5 seeds. The preregistered 5/5 threshold was not met, so the claim is
not upgraded.**

The budget below was declared before launch and was not changed.

## Declared budget

- 5 seeds: 0, 1, 2, 3, 4.
- 20 generations, 16 agents, 8 steps per generation, 14x14 world.
- Proprioception on, unit-flow feature, flow head.
- Min-alignment fitness, 6 frozen steps per direction.
- Rank replacement with `elite_fraction = 0.5`, immigrant fraction `0.25`.
- Displacement reward `5.0`, eval steps `12`.

Gate per held-out diagonal: evolved beats the initial population by at least
`0.2`, and the reversed-law alignment is at least `0.2`.

## Results

| Seed | Passed | Evolved minus initial | Minimum reversed-law |
| --- | --- | --- | --- |
| 0 | yes | +0.657 | +0.616 |
| 1 | yes | +0.283 | +0.480 |
| 2 | yes | +0.258 | +0.392 |
| 3 | yes | +0.368 | +0.539 |
| 4 | **no** (0.184 vs 0.200) | +0.184 | +0.598 |

**4/5 seeds pass.**

## What the failure is, and what it is not

Seed 4 misses the delta margin by `0.016`. Its reversed-law alignment is `+0.598`,
and its training alignment stays between `0.51` and `0.86`. It follows the law. It
does not beat the initial population by the full `0.2` on its worst held-out
diagonal.

So the failure is a margin miss, not a fixed action and not a collapse to noise.

## Compared with the 8-generation run

| Budget | Seeds passing | Failing seed | Failing delta |
| --- | --- | --- | --- |
| 8 generations | 4/5 | 1 | 0.193 |
| 20 generations | 4/5 | 4 | 0.184 |

The failing seed **changes** between the two runs, and both failures sit just below
the margin. That is consistent with one seed per five landing near the boundary, not
with one unlucky seed. The mechanism is stable; the `5/5` threshold is not met at
this budget.

All ten seed-runs across both budgets show the law-following signature: positive
reversed-law alignment and high, non-collapsing training alignment.

## Recording rule followed

The preregistered rule said: if 5/5 pass, upgrade the claim; otherwise report every
seed and do not retune. No retune was made and no budget was changed.

## Claim

The claim stays: **evolved embodied organisms behaviorally exploit the imposed law
in a bounded pilot, 4 of 5 seeds, confirmed at two budgets, under a four-parameter
movement head.** It is not a five-seed claim, and the `5/5` threshold remains open.

If a five-seed claim is required, the next action is a new preregistered run in a new
session with a stated change, for example more seeds or a stronger margin, not a
silent retune of this one.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 20 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --unit-flow-features --flow-head \
  --fitness-mode min_alignment --rank-replacement --immigrant-fraction 0.25 \
  --output docs/results/embodied_law_confirmation/summary.json
```
