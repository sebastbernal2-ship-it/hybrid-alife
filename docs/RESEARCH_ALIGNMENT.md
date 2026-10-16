# Research alignment

This document owns the original research question and the terms used by the implementation.
Other protocol documents must point here instead of changing the research target.

## Origin question

Create simple artificial organisms in a world with an imposed environmental condition, such as gravity, and observe what they do after exposure to that condition.

The central question is:

> Can an artificial organism discover or exploit a regularity in its world through interaction, even when it has no explicit concept or symbolic representation of that regularity?

The organism does not need to say what the condition is.
It does not need subjective awareness.
It does not need to solve a verbal puzzle.
Behavioral use of the condition is already a meaningful result.

## Levels of behavior

Keep these levels separate.

- Exposure: the organism experiences consequences of the environmental condition.
- Exploitation: the organism changes behavior in a way that improves outcomes under the condition.
- Prediction: the organism anticipates the condition's consequences in a new situation.
- Transfer: the organism uses the same regularity after a surface, strength, or orientation change.
- Adaptation: the organism changes behavior after the condition reverses or changes.
- Explicit representation: an internal state or communication channel decodes the condition.
- Consciousness: subjective experience.

The study's primary target is exploitation, prediction, transfer, and adaptation.
Explicit representation is optional.
Consciousness is outside the claim.

## Latent does not mean inaccessible

The condition is not hidden from interaction.
It is latent in the sense that the rule is not handed to the organism as a label.
The organism may observe movement, collisions, rewards, sensory effects, and other consequences.
Those consequences are valid evidence channels.

No observation leakage means that evaluator-only ground truth must not enter the controller through an undeclared channel.
It does not mean that the organism receives no evidence.

## Revelation conditions

A revelation is a declared information treatment, not an internal consciousness claim.
Test these conditions separately.

- No-revelation: the organism receives only ordinary environmental consequences.
- Demonstration: the world produces a salient event caused by the condition.
- Sensory hint: a new noisy signal correlates with the condition.
- Symbolic statement: an input explicitly states the condition or its consequence.
- Self-discovered state: the organism's learned internal state is tested for a change after experience.

A revelation must be logged with its timing, content, reliability, cost, and observation channel.
A symbolic statement must be labeled as an external message.
It must not be described as the organism's own realization.

## Gravity example

A first concrete world can impose a gravity field.
The organism can move, fall, collide, expend energy, and reproduce in that world.
The controller receives only the configured sensors and consequences.

The evaluation changes one factor at a time.

- Gravity present versus no gravity.
- Gravity strength changes.
- Gravity direction reverses.
- Surface texture changes while gravity stays fixed.
- Gravity-correlated cues are shuffled.
- Gravity is present but behavior cannot benefit from it.

The organism passes the behavioral test only when its improvement survives held-out strengths, directions, surfaces, and cue controls.

## Required controls

Use passive and active controls.

- Random policy.
- No-condition world.
- Shuffled-condition world.
- Cue-only policy.
- Extra-training control.
- Memory-disabled organism.
- Direct-observation upper bound.
- Scripted model-use oracle.
- False or misleading revelation.
- Revelation withheld after matched exposure.

The controls separate environmental exploitation from memorization, cue following, reward chasing, and extra computation.

## Required measurements

Record per-episode actions, observations, revelations, rewards, organism state summaries, and evaluator-only labels in separate fields.

Primary measurements are:

- Outcome improvement under the condition.
- Prediction accuracy on held-out consequences.
- Adaptation latency after condition reversal.
- Transfer across surface and strength changes.
- Response to a true revelation.
- Response to a false revelation.
- Difference between revealed and unrevealed conditions.
- Difference between ordinary exposure and matched extra training.

Do not infer an internal concept from behavior alone.
Use representation probes only as a separate optional analysis.

## Current project mapping

- `latent_law_task.py` is the first gravity-like latent-law calibration task.
- It logs no-revelation, demonstration, noisy-hint, symbolic-message, and no-benefit treatments.
- `causal_task.py` is an abstract hidden-rule calibration task.
- `causal_controller.py` is a controller-training calibration path.
- `run_causal_rollout.py` is a field-flow proxy, not the gravity task.
- PPO-POET archive controls are a supporting archive question.
- The latent-law task is calibration infrastructure, not learned-organism evidence.
- `latent_law_controller.py` now provides a bounded JAX policy-gradient calibration trainer.
- `scripts/run_gravity_pilot.py` runs the bounded held-out gravity pilot.
- None of these currently establishes the original organism-in-a-world result.

## Gravity pilot result (calibration only)

The training directions are the four cardinals.
The evaluation directions are four diagonals that never appear in training, so a constant action cannot pass.

The pilot measures movement alignment with the held-out gravity vector and total reward.

| Condition | Held-out reward |
| --- | --- |
| oracle (moves along gravity) | -22.88 |
| rest (does nothing) | -22.99 |
| random | -24.05 |
| constant move-down | -31.74 |
| learned controller | -23.85 |

The table above was produced by the first world design and is retained only as history.
That design was degenerate: the target sat at `unit_gravity * world_limit`, the organism had no braking action, and on a diagonal the organism was pushed past the target and pinned in the corner paying a permanent penalty no action could remove.
Rest scored `-22.99` and the best single action scored `-22.88`, a gap of `0.11`, so the diagonal hold-out tested nothing.

The world was redesigned to remove that flaw.
The target now sits at `target_radius = 2.0` inside a `6.0` world, so it is reachable and stoppable.
Action acceleration is `0.25`, above the `0.2` gravity magnitude, so the organism can brake and hold station.
Four diagonal thrust actions were added, for nine unit actions total, because an axis-only action set oscillates and cannot hold a diagonal station.

With the redesign, the scripted oracle (greedy one-step lookahead that knows gravity) holds station at `-18.26` for every direction.
Rest gives `-37.06`, random gives `-38.06`, and the constant move-down policy gives `-52.30`.
The oracle gap over rest is `+10.58` on cardinals and `+18.80` on diagonals, so both training and held-out directions carry a strict optimum.

## Gravity pilot on the redesigned world (calibration only)

Training uses the four cardinal directions and `NO_REVELATION` only.
Evaluation uses four held-out diagonals.

The learned controller reaches `-36.47`, which is only `0.59` above rest and far from the oracle.
Per direction it moves with gravity on three of four diagonals, so it learned to ride the flow, but it overshoots the target because it never learned to brake.
Training return moved only from `-33.13` to `-32.15` over `120` updates, and the critic loss barely moved, so the policy is far from converged.

The no-benefit control was tried both ways.
With `NO_BENEFIT` held out, the learned policy scores `-36.47`.
With `NO_BENEFIT` mixed into training, the learned policy scores `-50.06`, which is worse than resting.
Mixing the control into training hurts, because at step zero the no-benefit and gravity conditions look identical, so the policy becomes partly unobservable.

Status: the organism has learned direction-consistent movement but has not learned to reach and hold the target.
Gravity exploitation is therefore not demonstrated.
The benchmark is now non-degenerate, and the remaining gap is a learner and optimization problem.

## Detection metric and its replication

The movement rate is the fraction of steps that are not rest.
The detection gap is the movement rate with gravity present minus the movement rate with gravity absent.
A constant policy scores exactly zero, so the metric cannot be gamed by always moving or always resting.

A single seed first suggested a gap of `+0.46`.
Replication over five training seeds gave per-seed gaps of `+0.458`, `0.000`, `+0.458`, `0.000`, `0.000`, with mean `+0.183` and standard deviation `0.251`.
Three of five seeds show exactly zero.
The hint did not replicate, so law detection is not established.

## Conditioning of the control problem

Control authority is action acceleration divided by the gravity magnitude.

| Authority | Oracle reward | Final distance to target | Gap over rest |
| --- | --- | --- | --- |
| 1.25x | -18.26 | 2.248 | +10.58 |
| 1.50x | -14.17 | 0.881 | +14.67 |
| 2.00x | -9.35 | 0.066 | +19.49 |
| 2.50x | -8.62 | 0.158 | +20.22 |

At the original `1.25x` the organism cannot brake enough to hold the target, so the optimum demands near-perfect timing.
At `2.00x` the oracle holds the target to within `0.066` and the optimum is clean.

## Learner impasse

Even at `2.00x` authority the learner does not improve.
The checks below show the problem is the neural policy update, not the world.

- The plumbing is correct: forcing `move_down` gives `-38.46`, forcing `move_up` gives `-68.26`, and forcing `rest` gives `-28.84`, matching the hand-measured constant policies.
- The initial policy chooses `move_up`, the worst action.
- Six gradient steps at learning rate `0.5` leave the evaluated reward at exactly `-68.26`, so the deterministic action never changes.
- The same result appears at learning rates `0.02`, `0.005`, `0.002` and entropy coefficients `0.0`, `0.05`, `0.5`, `2.0`.
- Return variance is large, standard deviation `10.39` over 256 rollouts, so a learning signal exists.
- The initial action logits span only about `0.03`, so the deterministic action is decided by rounding noise and the update never moves it enough to change the choice.

Diagnosis, refined: both a recurrent actor-critic and a plain linear policy converge to resting or worse.
The root cause is the reward landscape, not the optimizer.

## The reward is a needle

Measured at `2.00x` authority on the single direction `(0, -0.2)`, horizon `12`.

Riding gravity and then resting:

| Ride steps | Total reward |
| --- | --- |
| 0 (pure rest) | -28.84 |
| 1 | -33.16 |
| 2 | -36.54 |
| 3 or more | -38.46 |

Every partial ride is worse than doing nothing, so there is no improvement path from rest toward the optimum.

Riding gravity and then braking upward:

| Ride steps before braking | Total reward |
| --- | --- |
| 0 | -68.26 |
| 1 | -35.91 |
| 2 | -11.94 |
| 3 | -34.66 |
| 4 or more | -38.46 |

One brake point is good and its immediate neighbours are worse by more than `23` points.
The optimum is an isolated spike in a discrete schedule space of twelve choices.

A mean-baseline policy-gradient method cannot find a spike like this, so the observed convergence to rest is the expected outcome rather than a learner bug.
The plumbing checks above still hold and the signal is still real, but the optimum is unreachable by incremental improvement.

## The displacement reward (adopted)

The position-target reward was replaced with a **displacement reward**: each step rewards movement along the gravity axis and penalises movement against it.

- Rest scores `0`, so it is neutral and no longer a trap.
- Moving with gravity scores positive, and each aligned step improves the total.
- Moving against gravity scores negative.
- The optimum is simply to move along gravity, which gives `+12` over `12` steps.

`GravityConfig.reward_mode` selects `"displacement"` (default) or `"target"`.
The target mode is retained only so the needle finding stays reproducible.

## First learned result on the displacement task (calibration only)

A linear policy over the encoded observation was trained on the four cardinal directions and evaluated on the four held-out diagonals.

Mean held-out reward over five seeds was `+3.04` with standard deviation `0.19`.
Every constant-action baseline scores exactly `0.00` on those symmetric diagonals, and the oracle scores `+12.00`.

WARNING: the mean is misleading and must not be reported as a headline.
The per-direction breakdown for one trained policy is:

| Held-out gravity | Reward | Actions taken |
| --- | --- | --- |
| (0.141, -0.141) | +8.49 | move right, all 12 steps |
| (0.141, 0.141) | +8.49 | move right, all 12 steps |
| (-0.141, 0.141) | +4.24 | move right 3 steps, then up 9 steps |
| (-0.141, -0.141) | -8.49 | move right, all 12 steps |

The policy detects gravity on one held-out direction and fails badly on another.
Its positive mean comes from a fixed rightward bias plus symmetric cancellation, not from reliable inference.
Averaging over antipodal directions hides a catastrophic per-direction failure, so per-direction results are the only sound report.

The recurrent actor-critic scores `0.00` on the same task, so the linear policy is the stronger learner here.

Status: gravity exploitation is still not demonstrated.
The task is now well posed, and the learner shows a small amount of genuine adaptation on one held-out direction.

The movement-rate detection metric belongs to the target and no-benefit framing.
Under the displacement reward the organism thrusts every step in both conditions, so the appropriate detection measure is an alignment difference, not a movement difference.

The learned controller ends with held-out alignment `-0.29`, meaning it moves slightly against gravity, and it scores worse than doing nothing.
Training return improved from `-21.74` to `-20.03`, so the update does move parameters, but the policy stays near-uniform and the deterministic action is close to random.

The organism has not demonstrated gravity exploitation.
The task is fair: gravity is recoverable from observed velocity drift plus the organism's own action, and the oracle beats rest.
The failure is in the learning method, not in the task contract.

The episode-boundary bug below was found and fixed during this pilot, and it is now guarded by a test.

## Final claim boundary

A known trainer defect found during the pilot:

- The recurrent loss ran one `lax.scan` over concatenated episodes, so hidden state leaked across episode boundaries.
- `_policy_loss` now resets hidden state per episode and is order-invariant, guarded by `test_training_loss_is_invariant_to_episode_order`.

The next required change is a stronger learner, such as a value-function critic with entropy control, not more objective shaping.

A positive result may support:

> Under the declared conditions, the artificial organisms behaviorally exploit and adapt to an imposed environmental regularity, including in some cases without an explicit symbolic representation.

A positive result may not support:

> The organisms understand the condition subjectively.

It may not support consciousness, sentience, or subjective experience.
