# Unit-flow normalization: results

Status: **negative pilot, normalization ruled out, selection is the limiter**.

## Question

The linear controller can express the flow mapping, but evolution cannot reach
the required weight scale because the raw flow features are about `0.01` and the
needed weights are about `100`. Does a scale-free, order-one flow feature make
the mapping reachable?

## Change

`EmbodiedConfig.unit_flow_features` (default off). When on, the observation
appends two features:

```
(previous_velocity - previous_thrust) / max(norm, eps)
```

This is the unit flow estimate: order one, scale-free, and independent of the
flow magnitude. The default observation is unchanged.

A hand-set test proves a linear controller with **order-one** weights on this
feature aligns above `0.99`. So the scale gap is removed.

## Protocol

- Primary: 5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
  proprioception on, linear controller, unit-flow feature, displacement reward
  `5.0`.
- Diagnostic, reported separately: one seed at 20 generations.
- Gate and evaluation unchanged: held-out diagonals, initial-policy control,
  corrected reversed-law control (`reversed >= 0.2`).

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Raw features, 8 generations (previous) | 0/5 | -0.48 | -0.65 |
| Unit flow, 8 generations | **0/5** | -0.25 | -0.72 |
| Unit flow, 20 generations, seed 0 | 0/1 | -0.36 | -0.71 |

The scale gap shrank. The failure did not.

## Diagnosis: the selection scheme is the limiter

Training alignment still oscillates with the sign of the imposed direction. In
the 20-generation run the late-generation alignment settles at about `+0.71` and
`-0.71`, the signature of a fixed action along a diagonal.

The reason is the fitness signal, not the observation. The displacement reward is
per-step and per-direction. Across the four training directions, a fixed action
gets a positive reward in the directions it matches and an equal negative reward
in the others, so its average is near zero. There is no consistent pressure to
condition on the flow, and selection uses total energy, where the alignment term
is small next to eating and basal costs. A flow-following policy and a fixed
action can therefore score similarly, and the population settles on whichever
fixed action the noise favored.

## What this implies

The stop condition is reached: with order-one features, evolution still fails, so
the next design must change **how fitness is assigned or how genomes are
explored**, not the observation.

The strongest candidate: fitness is the **minimum alignment across the four
training directions**, not total energy. A fixed action scores near `-1` on its
worst direction. A flow-following policy scores near `+1` on all of them. That
makes the law-conditioned mapping the only high-fitness behavior. Alternatives
are shared fitness across directions, or selection directly on alignment with a
variance penalty.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 \
  --displacement-reward 5.0 --linear-policy --unit-flow-features \
  --output docs/results/embodied_law_unit_flow/summary.json
```

Diagnostic: `diagnostics/generations_20.json`.
