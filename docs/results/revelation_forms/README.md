# Revelation forms: results

Status: **calibration only**.

## Question

Which forms of revelation help, does a false one get corrected, and does timing
matter?

## Protocol

- Law: the simulator's uniform flow field, horizon `12`.
- Policy: thrust = internal estimate + revelation when a revelation is present.
- Channels: exact, demonstration, noisy (sigma `0.15`), symbolic.
- Demonstration delivers the observed physical effect. In a uniform world it
  carries the same information as the exact vector, and the results confirm that.
- Symbolic names only the nearest cardinal axis, so it is genuinely lossy for
  diagonal laws.
- `message-only` is the control policy that uses the revelation and ignores the
  estimate.
- Directions are reported as a mean over six directions, three of them held-out
  diagonals. Per-direction detail lives in `summary.json`.

## Results

| Condition | Mean reward |
| --- | --- |
| no revelation (baseline) | +11.00 |
| true exact, early | +12.00 |
| true demonstration, early | +12.00 |
| true noisy, early | +11.94 |
| true symbolic, early | +11.80 |
| true exact, late | +11.00 |
| true demonstration, late | +11.00 |
| true noisy, late | +10.99 |
| true symbolic, late | +10.86 |
| false exact, early | +10.00 |
| false demonstration, early | +10.00 |
| false noisy, early | +10.09 |
| false symbolic, early | +10.20 |
| false exact, late | +9.00 |
| message-only, true | +1.00 |
| message-only, false | -1.00 |

## Interpretation

- An **early true** revelation helps in every form. The ordering follows the
  information the channel carries: exact and demonstration reach the full `+12`,
  noisy is close, symbolic is slightly lower because it quantises diagonals.
- A **late true** revelation adds nothing, because the internal estimate is
  already exact by then. A late symbolic message can even cost a little.
- A **false** revelation costs about one step early, and about two steps late,
  then the estimate corrects it. A lossy symbolic falsehood is less harmful than
  an exact one, which is consistent with it carrying less usable misleading
  information.
- A **message-only** policy is insufficient: `+1` at best, `-1` at worst. The
  revelation is useful only together with the internal estimate.

Claim boundary: these are declared external message channels. The study does not
show that the organism produces the realisation itself, and the numbers come from
a mechanism-isolating hand-built combination policy rather than a trained policy.
`test_trained_policy_can_use_a_revelation` covers the trained case for the exact
channel.

## Reproduction

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_revelation_study.py \
  --output docs/results/revelation_forms/summary.json
```
