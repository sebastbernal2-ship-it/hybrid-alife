# Flow-only movement head: results

Status: **positive bounded pilot, 4/5 seeds at the declared budget; the fifth seed
clears at 20 generations (reported separately).**

This is the first positive result for the evolved-embodied law claim. It is a
bounded pilot, not a five-seed held-out claim at the declared budget.

## Question

Every wider genome failed because the flow signal was one small term among roughly
a hundred weights: sparse credit. Does a four-parameter movement head let the
population learn the sign of the gain at all?

## Change

`EmbodiedConfig.flow_head` (default off). The genome is `w_flow` (n, 2, 2) and
`b_flow` (n, 2), and the movement raw is:

```
raw_move = einsum("nf,nfa->na", unit_flow, w_flow) + b_flow
```

Gates and messages are zero. The task is movement only. The head needs
`unit_flow_features`, which needs `proprioception`.

A hand-set `w_flow = I` reaches alignment above `0.99`.

## Protocol

5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
proprioception on, unit-flow feature, flow head, min-alignment fitness with 6
frozen steps per direction, rank replacement with `elite_fraction = 0.5`, 25%
random immigrants, displacement reward `5.0`.

Gate per held-out diagonal: evolved beats the initial population by `0.2`, and the
reversed-law alignment is at least `0.2`.

## Results

| Seed | Passed | Evolved minus initial | Minimum reversed-law |
| --- | --- | --- | --- |
| 0 | yes | +0.659 | +0.567 |
| 1 | **no** (0.193 vs 0.200) | +0.193 | +0.411 |
| 2 | yes | +0.482 | +0.615 |
| 3 | yes | +0.376 | +0.547 |
| 4 | yes | +0.210 | +0.576 |

**4/5 seeds pass at the declared 8-generation budget.**

Separate diagnostic, seed 1 at 20 generations: **passes**, evolved minus initial
`+0.283`, minimum reversed-law `+0.480`, and training alignment reaches about
`0.79` and stays high.

## Why this is law-following, not a fixed action

Two signatures separate the two:

- Training alignment now **trends upward**, for example seed 0:
  `0.33, 0.26, 0.30, 0.69, 0.56, 0.76, 0.49, 0.66`.
- The reversed-law alignment is **positive** in every seed (`+0.41` to `+0.62`).
  A fixed action flips sign under reversal. A law-follower re-infers the reversed
  flow and stays aligned with whatever law is imposed.

The earlier pilots failed exactly these two checks.

## What the full arc established

- The law is learnable when the movement decision is low-dimensional.
- The earlier failures were not the law being unlearnable. They were, in order:
  no birth or mutation, evaluation on the elite half only, an inverted reversed-law
  gate, feature scale, a per-episode advantage baseline, a no-op selection
  operator, and finally sparse credit.
- The four-parameter head isolates the sign of the gain, and the population learns
  it.

## Claim boundary

This is a bounded pilot. Do not call it `5/5`.

The next step for a full claim is one preregistered run: 5 seeds at 20
generations, declared before launch, with the same gate. If that passes 5/5, the
claim upgrades. If it does not, report per seed again.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --unit-flow-features --flow-head \
  --fitness-mode min_alignment --rank-replacement --immigrant-fraction 0.25 \
  --output docs/results/embodied_law_flow_head/summary.json
```

Diagnostic: `diagnostics/seed1_generations_20.json`.
