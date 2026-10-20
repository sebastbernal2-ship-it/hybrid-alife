# Policy-gradient update: results

Status: **negative pilot, lever 1 ruled out as stated**.

## Question

Evolution and survivor mutation both fail and the population carries a fixed
action bias. Can a gradient update on the embodied controller move the policy to
a flow-conditioned one where random mutation did not?

## What was built

`src/hybrid_alife/experiments/latent_law_embodied_pg.py`, following the
point-organism trainer in `latent_law_continuous.py`:

- REINFORCE on a Gaussian movement mean. The two movement means receive
  exploration noise and a score-function gradient; gates and messages stay
  deterministic.
- Collection runs the episode, then a replayed log-prob loss recomputes the
  Gaussian log-probability from the stored observations and hidden states. The
  gradient flows only through the current mean.
- Training reward is the same alignment signal the gate measures:
  `dot(movement, unit_law)`. Free advection is excluded.
- `scripts/run_embodied_policy_gradient.py` trains on the four cardinals and
  evaluates frozen on the held-out diagonals, with the initial-policy and
  reversed-law controls.

## Protocol

Primary run: 5 seeds, 60 updates, 8 episodes per update, 8 steps, learning rate
`0.05`, exploration `sigma=0.4`, one unscored warm-up step so the first scored
observation already carries the flow in `previous_velocity - previous_thrust`.

Diagnostics: no warm-up, and random training directions.

## Results

| Run | Seeds passing | Worst evolved minus initial | Worst reversed-law |
| --- | --- | --- | --- |
| Warm-up, cardinals, 5 seeds | **0/5** | -1.90 | -0.99 |
| No warm-up, cardinals, 5 seeds | 0/5 | -1.54 | -0.93 |
| Warm-up, random directions, seed 0 | 0/1 | -1.30 | -0.95 |

The gradient does move the policy. The single-direction test moves held-out
alignment up by more than `0.1`, and trained magnitudes reach `0.99`. The
deterministic mean is still a **fixed action**, not a flow-conditioned one:

- trained alignment is antisymmetric under direction negation
  (`a(d) = -a(-d)`) in every seed, and
- the reversed-law alignment is the negation of the trained alignment, not a
  positive re-inference.

So the policy learned a direction, not the law.

## Two bugs found and fixed

1. **The reversed-law gate direction was inverted** in the first two write-ups.
   A law-following policy re-infers the flow and still aligns with the imposed
   law after reversal, so its reversed-law alignment stays **positive**. A
   fixed-action policy flips sign. The correct condition is
   `reversed >= 0.2`. The earlier verdicts are unchanged: their minimum reversed
   alignment is negative (-0.14 to -0.39 for proprioception, -0.10 to -0.31 for
   survivor mutation), so both controls fail under either reading.
2. **The advantage baseline was per episode.** The point-organism trainer
   baselines across episodes at each timestep. A per-episode baseline removes the
   cross-episode signal that separates a fixed action (positive in half the
   episodes, negative in half) from a flow-follower (positive in all of them).
   Corrected to the across-episode baseline. The fixed-action collapse remained.

## Diagnosis

The gradient reaches a fixed action because a direct bias path can satisfy the
current episode, and the credit assigned to the proprioceptive features is weak
relative to that path. Warm-up, random training directions, and the corrected
baseline each remove a shortcut, but none produced a flow-conditioned mean. The
remaining question is representational: can this controller express the
inferred-flow mapping at all, and can a gradient find it? That is lever 3.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_policy_gradient.py \
  --seeds 0 1 2 3 4 --updates 60 --episodes-per-update 8 \
  --steps 8 --size 14 --eval-steps 12 \
  --learning-rate 0.05 --sigma 0.4 \
  --output docs/results/embodied_law_policy_gradient/summary.json
```

Diagnostics: `diagnostics/no_warmup.json`, `diagnostics/random_directions.json`.
