# Expressive controller: results

Status: **negative pilot, lever 3 ruled out, search is the limiter**.

## Question

The policy gradient moves the embodied policy but converges to a fixed action. Is
the limiter the controller's ability to represent the inferred-flow mapping, or
the search?

## Change

`EmbodiedConfig.linear_policy` (default off). When on, the GRU is replaced by a
flat linear map, `raw = einstein("no,noa->na", obs, w_linear) + b_linear`, with a
genome of `w_linear` and `b_linear`. The action layout is unchanged, so
`decode_actions` still applies. This controller can express the flow mapping
exactly:

- `previous_velocity - previous_thrust` is proportional to the local flow, and
- a linear row that reads those features and writes the movement mean follows
  the flow.

A hand-set test proves it: with the flow features mapped to the movement pair,
the deterministic action alignment exceeds `0.99`.

## Protocol

5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
proprioception on, displacement reward `5.0`, linear controller, evolution on the
four cardinals, frozen evaluation on the held-out diagonals, initial-policy and
corrected reversed-law controls.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Linear controller, 5 seeds | **0/5** | -0.48 | -0.65 |

Per seed: `min_delta` is -0.484, -0.268, -0.416, -0.225, -0.219. Minimum
reversed-law alignment is -0.651, -0.529, -0.328, -0.363, -0.339.

Training alignment still oscillates with the sign of the imposed direction. The
population still carries a fixed action bias.

## Diagnosis: the representation is sufficient, the search is not

The controller can express the mapping, so representation is not the limiter.
The scale is:

- The flow features are `previous_velocity ~ flow * flow_movement_scale`, with
  `flow_movement_scale = 0.05` and a flow magnitude of `0.2`, so the features are
  about `0.01`.
- To turn `0.01` into an action near `1`, the linear weights must be about `100`.
- Genomes initialize at `0.1`, mutation has standard deviation `0.05` and
  probability `0.1`, and the run has 8 generations. Mutation cannot bridge a
  three-order-of-magnitude gap.

So evolution explores a neighborhood around the initial scale and cannot reach
the weights the task needs. This is a search-and-scaling limit, not an
architectural one.

## What this implies

The claim needs a different study design, not another architecture swap. Two
concrete options:

1. **Normalize the proprioceptive features**, for example append
   `(previous_velocity - previous_thrust) / flow_movement_scale`, so the flow
   signal is order one and the needed weights are order one.
2. **Initialize at the task scale**, or give the controller a learnable per-feature
   scale, so a local mutation can reach the useful region.

Either option changes the observation or the initialization and invalidates
checkpoints. State it as a new package and rerun the same gate.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 \
  --displacement-reward 5.0 --linear-policy \
  --output docs/results/embodied_law_linear_policy/summary.json
```
