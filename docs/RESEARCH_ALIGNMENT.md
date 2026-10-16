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
