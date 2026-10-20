# Survivor mutation: results

Status: **negative pilot, lever ruled out**.

## Question

The D1 proprioception pilot failed and selection only mutated dead slots, so a
dominating clone froze the population. Does mutating survivors every generation
let the population escape the fixed action bias and learn the law?

## Change

`EmbodiedConfig.mutate_survivors` (default off). When on, each generation applies
the existing mutation mask and noise to every genome. Survivors keep their own
genome plus noise; dead slots still take a mutated tournament parent. Three tests
in `tests/test_embodied_selection.py` cover the default freeze, the enabled
mutation, and the perturbation size.

## Protocol

Same as the proprioception run, with survivor mutation on:

- 5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world.
- Displacement reward `5.0`, proprioception on.
- Held-out diagonals, initial-population control, reversed-law control.
- Gate per direction: evolved beats initial by `0.2`, and reversed-law is at
  least `0.2`. A law-follower re-infers the reversed flow and stays aligned; a
  fixed action flips sign. See the correction note.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Frozen elite, reward 5.0 | 0/5 | -0.26 | -0.39 |
| Survivor mutation, reward 5.0 | **0/5** | -0.35 | -0.31 |

Per seed: `min_delta` is -0.095, -0.216, -0.198, -0.349, -0.220. The minimum
reversed alignment is -0.262, -0.236, -0.311, -0.305, -0.104.

Survivor mutation moved the numbers but not the sign. No seed beats the initial
control on the worst direction, and no seed reaches the reversed-law threshold:
the minimum reversed alignment is negative in every seed.

Training alignment still tracks the sign of the imposed direction:

| Generation | Imposed direction | Training alignment |
| --- | --- | --- |
| 0 | north | -0.01 |
| 1 | south | +0.22 |
| 2 | east | +0.40 |
| 3 | west | -0.31 |
| 4 | north | -0.17 |
| 5 | south | +0.21 |

That is the same fixed action bias pattern. The population still carries a fixed
direction and does not condition on the inferred flow.

## Diagnosis

The frozen elite was **not** the primary blocker. Mutating survivors changes the
search path but the population still fails to find a flow-conditioned policy
within this budget. Two levers remain:

1. A policy-gradient update on the embodied controller, which can move the policy
   where random mutation does not.
2. An expressive controller that can represent the inferred-flow mapping
   directly, to test whether the weights are reachable at all.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 \
  --displacement-reward 5.0 --mutate-survivors \
  --output docs/results/embodied_law_survivor_mutation/summary.json
```

## Correction: reversed-law control direction

An earlier version required the reversed-law alignment to be at most `0.0`. That
is backwards. A law-follower re-infers the reversed flow and stays positive under
reversal; a fixed action flips sign. The corrected condition is `reversed >= 0.2`,
and the table reports the minimum reversed alignment. Every seed still fails. The
verdict is unchanged. Full analysis in
`docs/results/embodied_law_policy_gradient/README.md`.
