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
- Its first smoke remains calibration-only and does not show revelation use or gravity exploitation.
- None of these currently establishes the original organism-in-a-world result.

## Final claim boundary

A positive result may support:

> Under the declared conditions, the artificial organisms behaviorally exploit and adapt to an imposed environmental regularity, including in some cases without an explicit symbolic representation.

A positive result may not support:

> The organisms understand the condition subjectively.

It may not support consciousness, sentience, or subjective experience.
