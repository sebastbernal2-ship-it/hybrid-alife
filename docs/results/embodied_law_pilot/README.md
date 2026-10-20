# Evolved embodied law pilot: results

Status: **negative pilot, calibration only**.

## Question

Do the actual evolved embodied organisms exploit an imposed law, rather than a
simplified point organism with a linear policy?

## Protocol

- Law: the simulator's uniform flow field, imposed each generation.
- The flow is hidden from the embodied observation (`hidden_field_groups` includes
  flow and every flow-derived channel), but it still couples physically through
  `flow_movement_scale` and is rewarded through `flow_alignment_reward`.
- Training alternates the four cardinal directions across generations.
- Evaluation is frozen (no mutation, no reproduction, Avida disabled) on four
  held-out diagonal directions, and every slot in the population is alive during
  evaluation.
- Control: the same evaluation on the **initial** population, which measures
  whether evolution improved anything at all.
- Budget: 16 agents, 14x14 world, 8 steps per generation, 8 or 20 generations.

## Results

| Generations | Evolved mean alignment | Initial mean alignment | Delta |
| --- | --- | --- | --- |
| 8 | +0.001 | +0.008 | -0.007 |
| 20 | +0.001 | +0.008 | -0.007 |

Per-direction held-out alignment stays between about `-0.12` and `+0.12` in both
cases. Training-time alignment oscillates around zero and does not trend upward.

So evolution changes the genomes but does **not** improve alignment with the law.
Law exploitation is not demonstrated in this pilot.

## Two plumbing defects found and fixed

1. With every slot alive and initial energy below the reproduction threshold, no
   births occurred, so nothing was ever mutated and the genomes never changed.
   Fixed by enabling reproduction and leaving dead slots.
2. The evaluation re-created a half-alive target population, so it measured only
   the unchanged elite slots and reported identical numbers for evolved and
   initial policies. Fixed by evaluating with every slot alive.

Both defects silently produced plausible-looking zeros. They are the reason the
first D1 readings looked like "no change" rather than "no effect".

## Why this is negative, and what would change it

- The embodied observation has no previous-velocity or previous-thrust features,
  so the organism cannot compute the law from its own motion the way the internal
  estimate does. A hidden law is therefore close to unlearnable from a small
  energy bonus alone.
- The alignment reward is a small per-step energy term, not the dominant fitness
  signal, so selection pressure toward alignment is weak.
- The budget is tiny: a few generations with 16 agents.

Next required step for D1: add explicit proprioceptive features to the embodied
observation, add a displacement-style reward that actually drives selection, and
only then run a larger budget with the same held-out gate. Until that runs, the
evolved-embodied claim stays open and must be reported as unproven.

## Reproduction

There is no standalone script yet; the pilot is driven by
`run_embodied_law_pilot` in `src/hybrid_alife/experiments/latent_law_embodied.py`.
