# Random immigrants: results

Status: **negative pilot, and it identifies a sparse-credit problem**.

## Question

Rank replacement keeps the least-negative elite, and local mutation never flips
the sign of the unit-flow gain. Does injecting fresh random genomes into the bottom
slots let the population reach an aligned basin?

## Change

`EmbodiedConfig.immigrant_fraction` (default `0.0`). When rank replacement is on,
the bottom `round(fraction * n)` slots receive fresh genomes from
`initialize_embodied_genomes` instead of mutated elite copies. The elite and the
mutated-middle slots behave as before. `select_and_mutate_embodied` takes an
optional `world_cfg`; immigrants raise a clear error if it is missing.

## Protocol

5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
proprioception on, linear controller, unit-flow feature, min-alignment fitness with
6 frozen steps per direction, rank replacement, immigrant fraction `0.25`
(4 fresh of 16 each generation), displacement reward `5.0`.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Rank replacement, no immigrants | 0/5 | -0.96 | -1.00 |
| Rank replacement, 25% immigrants | **0/5** | -0.73 | -0.66 |

Per seed: `min_delta` is -0.295, -0.350, -0.245, -0.730, -0.626. Minimum
reversed-law alignment is -0.284, -0.300, -0.203, -0.407, -0.662.

Immigrants help at the margin. They do not produce the mapping.

## Diagnosis: the flow signal is buried in the other weights

A diagnostic prints the best genome each generation:

| Generation | Best fitness | Unit-flow gain | Bias | Other-weight norm |
| --- | --- | --- | --- | --- |
| 0-2 | -0.60 | -0.18 | 0.00 | 4.47 |
| 3-7 | -0.71 | +0.04 | 0.00 | 4.50 |

The unit-flow gain never grows beyond about `0.04`, while the weights on the other
observation features have norm about `4.5`. The movement is
`tanh(w @ obs)`, so the other features dominate and the flow contribution is a
small term in a noisy sum. To align, a genome must zero roughly a hundred other
weights and grow the flow gain. Dense local mutation cannot do that in this budget.

So the fitness is correct, the operator is correct, and exploration helps. The
problem is **sparse credit**: the useful signal is a few weights inside a large
dense genome.

## Next design: a flow-only movement head

Make the movement head low-dimensional and let the rest of the genome handle only
the gates and messages, or add a strong weight penalty on the non-flow movement
weights.

The smallest decisive test: movement raw is
`w_flow @ [unit_flow_x, unit_flow_y] + b_flow`, a 2x2 matrix plus a 2-vector. Then
the search is four-dimensional, and the question is only whether the population
can learn the sign of the gain. If it cannot, the fitness or the selection is
broken. If it can, the full controller needs a sparsity mechanism, not more
mutation.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --linear-policy --unit-flow-features \
  --fitness-mode min_alignment --rank-replacement --immigrant-fraction 0.25 \
  --output docs/results/embodied_law_immigrants/summary.json
```
