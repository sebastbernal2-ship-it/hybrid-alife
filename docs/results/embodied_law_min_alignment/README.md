# Min-alignment fitness: results

Status: **negative pilot, and it exposes a no-op selection operator**.

## Question

Evolution settles on a fixed diagonal because a per-direction reward averages to
zero across directions and selection uses total energy. Does selecting directly
on the **minimum alignment across the four training directions** make the
flow-conditioned mapping the only high-fitness behavior?

## Change

`minimum_direction_alignment` evaluates every genome on all training directions
with a short frozen rollout and returns the minimum per organism. A fixed action
scores near `-1` on its worst direction; a flow-follower scores near `+1` on all
of them. `run_embodied_law_pilot` gains `fitness_mode="min_alignment"` and
`fitness_steps`.
A unit test proves the property: a hand-set flow-follower scores above `0.8` and
a hand-set fixed action below `-0.5`, with a gap above `1.0`.

## Protocol

5 seeds, 8 generations, 16 agents, 8 steps per generation, 14x14 world,
proprioception on, linear controller, unit-flow feature, displacement reward
`5.0`, min-alignment fitness with 6 frozen steps per direction, gate unchanged.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst (minimum) reversed-law |
| --- | --- | --- | --- |
| Min-alignment fitness, 5 seeds | **0/5** | -0.22 | -0.53 |

Per seed: `min_delta` is -0.217, -0.062, -0.222, -0.223, -0.159. Minimum
reversed-law alignment is -0.533, -0.096, -0.244, -0.480, -0.195.

## The decisive evidence

The training-alignment trajectories are **identical** to the unit-flow energy
run for seeds 0 and 3, generation by generation. For example, seed 0:

```
energy:        [0.20, 0.09, 0.15, -0.43, -0.39, 0.33, 0.34, -0.35]
min-alignment: [0.20, 0.09, 0.15, -0.43, -0.39, 0.33, 0.34, -0.35]
```

A different fitness cannot produce an identical population trajectory. The
fitness never reached the population.

## Diagnosis: selection only fills dead slots

`select_and_mutate_embodied` computes `dead = ~pop.alive` and writes
`jnp.where(dead, mutated_child, current_genome)`. When every slot is alive, it
leaves every genome unchanged, regardless of the fitness passed in.

Reproduction runs before selection and revives dead slots, so by the time
selection runs there can be no dead slots left to replace. The operator is a
no-op on a living population. The existing `test_survivors_are_frozen_by_default`
already documents this behavior.

So neither min-alignment fitness nor any earlier fitness change could act. The
limiter is not the fitness function; it is that **no living low-fitness organism
is ever replaced**.

## Next design: generational replacement by fitness rank

Replace survivor selection with a real operator:

1. Sort slots by fitness.
2. Keep the top `elite_fraction`.
3. Replace the rest with mutated copies of the top slots.

That acts on every generation, with or without deaths, and is the smallest change
that makes fitness matter. Keep the same gate and state the budget first.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --linear-policy --unit-flow-features \
  --fitness-mode min_alignment \
  --output docs/results/embodied_law_min_alignment/summary.json
```
