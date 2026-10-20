# Gravity study v1: results

Status: **calibration only**.
This document states what the study measured, what it supports, and what it does not.

## Question

Can a simple artificial organism, exposed to an imposed world condition such as
gravity, discover or behaviourally exploit that condition, including on
directions it never trained on?

It does not need to name the condition and it does not need an explicit symbol
for it. Behavioural exploitation is the target.

## Provenance

- Artifact: `docs/results/gravity_study_v1/summary.json`
- Checksums: `docs/results/gravity_study_v1/SHA256SUMS`
- Protocol owner: `docs/RESEARCH_ALIGNMENT.md`
- Reproduce with:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py \
  --seeds 0 1 2 3 4 \
  --output docs/results/gravity_study_v1/summary.json
```

## Protocol

- Law: a constant imposed force, a gravity-like law with magnitude `0.2`.
- Reward: displacement, `dot(unit_thrust, unit_law)`, in `[-1, +1]` per step.
  Rest scores `0`, thrusting along the law scores `+1`, against it `-1`.
- Horizon: `12` steps.
- Held-out rule: training never sees the four 45 degree diagonal directions.
  Every reported diagonal number is transfer to unseen directions.
- Learner: reward-weighted regression over a linear thrust policy, with
  exploration annealed from `0.6` to `0.05`.
- Gate: a seed passes only when the **worst** held-out direction is above `9.0`
  and the mean across directions is above `10.0`. The gate is a minimum across
  directions, so a fixed-action bias cannot pass.

## Findings

### 1. Calibration world, end to end

A linear policy learns the whole sensor-to-thrust mapping from scratch.

| Metric | Value |
| --- | --- |
| Seeds passing | 5 of 5 |
| Worst-direction mean | +9.94 +/- 0.13 |
| Worst single direction | +9.76 |
| Mean across directions | +10.85 |

Every constant-action policy scores `0.00` on those symmetric diagonals.

### 2. Real simulator, internal estimate

The law lives in the simulator's own `flow` field, and the world is stepped by
the real `step_world`. The organism forms an explicit estimate from
proprioception:

```text
law_estimate = velocity - friction * previous_velocity - acceleration * previous_thrust
```

The estimate uses only quantities the organism measured itself. It never sees
the world flow or any evaluator label.

| Metric | Value |
| --- | --- |
| Seeds passing | 5 of 5 |
| Worst-direction mean | +10.06 +/- 0.10 |
| Worst single direction | +9.90 |
| Mean across directions | +10.92 |

Best achievable is `+11`, because step zero has no history to estimate from.
The evaluator oracle scores `+12` because it already knows the law at step zero.

A control that ignores the estimate and uses only a constant bias scores
positive on some diagonals and negative on others, so the behaviour depends on
the internal estimate.

### 3. Revelations

A revelation is a declared external message channel.
A true revelation at step zero removes the step-zero blind spot.

| Treatment | Mean reward |
| --- | --- |
| no revelation | +11.00 |
| true revelation at step zero | +12.00 |
| false revelation, opposite direction | +10.00 |

A true revelation is used. A false one costs exactly one step, then the
organism's own estimate overrides it. The organism does not blindly trust the
channel.

## What this supports

- In this calibration world, an artificial organism behaviourally exploits an
  imposed gravity-like law through exposure and generalises it to held-out
  directions at five of five seeds.
- With a proprioceptive forward model, the same holds reliably inside the real
  simulator world.
- A correct revelation is exploited, and a false revelation is self-corrected.

## What this does not support

- Biology, or the evolved embodied GRU controller inside the shared simulator.
- The PPO-POET archive question.
- That the organism discovers the inference rule itself. In finding 2 the rule
  is hand-declared, like an efference-copy forward model, and only the policy on
  top of it is learned. Finding 1 is the end-to-end claim and it belongs to the
  calibration world.
- Symbolic representation, consciousness, sentience, or subjective experience.
- Generality beyond a constant two-dimensional force and a point organism.

## Rejected alternatives, for the record

- Position-target reward: rejected because its optimum is an isolated spike with
  no improvement path, so no incremental learner can find it.
- Nine discrete thrust actions: rejected because the learned mixture's mean is
  good and its mode is arbitrary, a mode-versus-mean failure.
- Plain policy gradient on continuous thrust: rejected because it collapses to a
  fixed direction.
- Feature scaling and more training data for the simulator end-to-end fit: both
  tested and both rejected, because neither changed the per-direction reliability.
