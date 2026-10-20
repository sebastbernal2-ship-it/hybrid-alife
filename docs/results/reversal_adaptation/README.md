# Reversal adaptation: results

Status: **calibration only**.

## Question

When the imposed law reverses mid-episode, does the organism adapt, and how fast?

## Protocol

- Law: the simulator's uniform flow field, reversed at step `8` of a `16` step episode.
- The organism is **not told** about the reversal. It can only detect the change from proprioception.
- Conditions: adaptive internal estimate, frozen estimate (memory-disabled control), and no reversal (control).
- Recovery: alignment at or above `0.8` for two consecutive post-reversal actions.
- Step zero and the reversal step are excluded from the alignment means. Step zero has no history and the reversal action was chosen before the flip.
- Directions are reported separately, never averaged across antipodes.

## Results

Adaptive internal estimate, five seeds per direction:

| Direction | Pre-alignment | Post-alignment | Recovered all seeds | Recovery latency | Regret |
| --- | --- | --- | --- | --- | --- |
| (0, -1) | +1.00 | +1.00 | yes | 1 | 2.00 |
| (0, +1) | +1.00 | +1.00 | yes | 1 | 2.00 |
| (+1, 0) | +1.00 | +1.00 | yes | 1 | 2.00 |
| (-1, 0) | +1.00 | +1.00 | yes | 1 | 2.00 |

Controls:

| Condition | Post-alignment | Recovered | Regret |
| --- | --- | --- | --- |
| adaptive estimate | +1.00 | all seeds | 2.00 |
| frozen estimate | -1.00 | none | 16.00 |
| no reversal | +1.00 | all seeds | 0.00 |

A trained estimate policy reproduces this: post-alignment minimum `+0.99`, frozen control maximum `-0.99`, maximum recovery latency `1`.

## Interpretation

The organism re-aligns in exactly one step after the law reverses, and a control that freezes its estimate never adapts at all, going fully anti-aligned.

That one-step cost is structural: the action at the reversal step was chosen before the flip. It is a real cost and is counted in regret.

Claim boundary: because the internal estimate is exact after a single transition, this shows fast **tracking** with a proprioceptive estimate. It is not evidence of learned, slowly-acquired reversal strategy. That stronger claim would need the estimate itself to be learned, which is the evolved-embodied direction.

## Reproduction

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_reversal_adaptation.py \
  --output docs/results/reversal_adaptation/summary.json
```
