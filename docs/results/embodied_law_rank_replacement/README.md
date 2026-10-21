# Rank replacement: results

Status: **negative pilot, and it exposes a local optimum in exploration**.

## Question

The min-alignment fitness is correct but never acted because selection only filled
dead slots. Does generational replacement by fitness rank make it act, and does
the population then learn the flow mapping?

## Change

`EmbodiedConfig.rank_replacement` (default off). When on, selection sorts slots by
fitness, keeps the top `elite_fraction` untouched (`0.5`, so 8 of 16), and replaces
the rest with mutated copies of the elite. Replaced slots get reset energy, age,
hidden state, messages, and previous-velocity fields.

The flag is threaded through the pilot, and a test asserts the pilot config carries
it. That test was added after a wiring bug: the first attempt silently fell back to
the default because the pilot's config call dropped the flag, so the run reproduced
the previous result exactly.

## Protocol

5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
proprioception on, linear controller, unit-flow feature, min-alignment fitness with
6 frozen steps per direction, rank replacement, displacement reward `5.0`.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Min-alignment, no rank replacement | 0/5 | -0.22 | -0.53 |
| Min-alignment, rank replacement | **0/5** | -0.96 | -1.00 |

Per seed: `min_delta` is -0.905, -0.892, -0.704, -0.550, -0.964. Minimum
reversed-law alignment is -1.000, -0.994, -0.726, -0.808, -1.000.

The operator acts: the run differs from the previous one. The outcome is worse, not
better.

## Diagnosis: a local optimum the mutation cannot escape

A diagnostic on one seed prints the per-slot fitness and the elite's unit-flow
weights each generation:

- Every slot's minimum alignment is **negative** from generation 0
  (`-0.60` to `-0.99`).
- The top slots have a **negative** unit-flow gain (about `-0.18`): they move
  against the flow.
- Bias is near zero, so the unit-flow term drives the behavior.
- Rank replacement keeps the least-negative elite, and local mutation never flips
  the sign of the unit gain.

So the fitness is correct, and the operator is correct, but the population starts in
an anti-aligned basin and cannot leave it. The hand-set test still shows a
flow-follower would score `+0.8`; one never appears.

## Next design: exploration

The mutation operator is the limiter, not the fitness or the selection.

1. Add random immigrants: replace the bottom slots with freshly initialized
   genomes instead of mutated elite copies, every few generations.
2. Or raise `genome_mutation_std` and the mutation probability so a sign flip is
   reachable.
3. Or initialize the unit-flow weight block positive, so the population starts
   aligned.

Keep the same gate and state the budget first. Prefer immigrants or a larger
mutation, because both are generic and do not encode the answer in the
initialization.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --linear-policy --unit-flow-features \
  --fitness-mode min_alignment --rank-replacement \
  --output docs/results/embodied_law_rank_replacement/summary.json
```
