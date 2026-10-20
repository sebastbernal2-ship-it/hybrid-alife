# Evolved embodied law with proprioception: results

Status: **negative pilot, calibration only**.

## Question

Can the actual evolved embodied organisms exploit an imposed law when their own
motion makes the law inferable?

The previous pilot failed because the embodied observation had no previous
velocity or previous thrust, so a hidden law could not be inferred from the
organism's movement. This package adds those features and a displacement reward,
then repeats the bounded pilot with the same controls.

## Changes

- `EmbodiedConfig.proprioception` (default off) appends the previous actual
  movement and the previous commanded thrust to the observation, four features
  in total. The default shape is unchanged.
- `EmbodiedConfig.displacement_reward` (default `0.0`) adds
  `reward * dot(commanded_displacement, unit_law)` to energy. The free advection
  term is excluded because it is identical for every organism and would only
  inflate energy without changing selection.
- `evaluate_frozen_alignment` and `per_direction_frozen_alignment` accept
  `reverse_law=True` for the reversed-law control.
- New script `scripts/run_embodied_law_study.py`.

## Protocol

- Training alternates the four cardinal directions across generations.
- Evaluation is frozen on the four held-out diagonals, with every slot alive.
- Controls: the initial population, and the same held-out directions with the
  law reversed.
- Gate, per direction: evolved alignment must beat the initial population by at
  least `0.2`, and the reversed-law alignment must be at most `0.0`.
- Primary run: 5 seeds, 8 generations, 16 agents, 8 steps per generation,
  14x14 world, displacement reward `5.0`.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst reversed-law |
| --- | --- | --- | --- |
| Primary, reward 5.0 | **0/5** | -0.26 | +0.40 |
| Reward 50.0, seed 0 | 0/1 | -0.21 | +0.41 |
| 20 generations, reward 50.0, seed 0 | 0/1 | -0.21 | +0.41 |

The gate fails in every configuration. Raising the reward tenfold and doubling
the generations changed nothing. The held-out values after 20 generations are
identical to those after 8, because selection only mutates dead slots, so the
surviving elite is frozen once it dominates.

## Diagnosis

Training alignment oscillates and tracks the sign of the imposed direction, not a
learned response to it:

| Generation | Imposed direction | Training alignment |
| --- | --- | --- |
| 0 | north | +0.11 |
| 1 | south | +0.31 |
| 2 | east | +0.43 |
| 3 | west | -0.41 |
| 4 | north | -0.40 |
| 5 | south | +0.39 |

A fixed action bias produces exactly this pattern: the cosine with the law flips
sign when the law flips. So the population is not following the law. It carries a
fixed action bias, and the gate correctly refuses it.

The reward plumbing is correct: a unit test shows the energy difference equals
`reward * 0.05` for a commanded step along the law. The limitation is
evolutionary search. With 16 agents, a GRU controller, and selection that leaves
survivors untouched, the population does not find a flow-conditioned policy
within this budget.

## What would change the result

- Mutate survivors as well as dead slots, or use a policy-gradient update on the
  embodied controller, so a better policy can spread.
- Use a much larger population and generation budget, and state the budget before
  running.
- Reduce the controller to a form that can express the inferred-flow mapping
  directly, then test whether evolution can find its weights.

Until one of those runs, the evolved-embodied claim stays open and unproven. This
is a bounded pilot, not a five-seed held-out claim.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 8 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 \
  --displacement-reward 5.0 \
  --output docs/results/embodied_law_proprioception/summary.json
```

Diagnostics: `diagnostics/reward_50.json`, `diagnostics/generations_20.json`.
