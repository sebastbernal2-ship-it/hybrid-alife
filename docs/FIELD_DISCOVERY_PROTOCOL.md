# Field Discovery Protocol

## Research question

Can an embodied artificial organism discover and exploit structured environmental fields through sensorimotor interaction?

The protocol measures functional environmental awareness.
It does not claim to detect subjective consciousness.

## Fields

Use the existing flow, curvature, enrichment, lift, concentration, and crowding fields.
Keep world dynamics identical across observation conditions.

The organism may receive direct field values, indirect consequences, or corrupted field cues.

## Conditions

### Direct access

The organism receives the selected field in its local observation patch and sixth-sense channels.

### Indirect access

Hide the selected field channels.
Keep resources, hazards, energy, movement, and chemical consequences active.

This condition tests whether the organism can learn the field from its effects.

### Decorrelated control

Keep actual flow dynamics and movement coupling active.
Translate observed flow vectors by half the toroidal world in both dimensions.
This roll preserves the complete flow-value distribution while moving values to different outcome locations.
Hide flow-derived shear and lift cues so they cannot leak the original location.

This is a spatially decorrelated flow observation, not a random shuffle of sixth-sense channels.
It tests whether performance depends on where flow occurs rather than its marginal values.

### No-field control

Set flow-driven movement coupling to zero.
Hide flow, shear, shear-gradient, and lift observations.
Keep flow transport and all other world dynamics active.
This isolates the embodied movement and observation contribution without disabling unrelated simulator effects.

### Intervention and transfer

Reverse the flow vector at the midpoint of frozen-policy evaluation.
Also evaluate the frozen policy in a separate rollout with flow strength multiplied by 1.5.
Do not adapt the controller during either evaluation.

The direction reversal tests response to a changed relationship.
The held-out strength tests transfer beyond the training field magnitude.

## Outcome levels

- **No use:** field access does not improve survival, energy, reproduction, or resource collection.
- **Incidental use:** performance improves in the training distribution but fails after intervention.
- **Implicit use:** the organism exploits the field without direct field observations.
- **Functional model:** the organism predicts consequences, transfers across field changes, and adapts faster than decorrelated controls.
- **Counterfactual flow information gain:** mutual information between flow magnitude and enrichment, minus mutual information between half-world-translated flow magnitude and enrichment.
This measures spatial flow-outcome correspondence against a same-value-distribution counterfactual.
It is an observational diagnostic, not an active-probing measure or proof of internal knowledge.

These levels describe behavior.
They do not establish subjective experience.

## Metrics

Report median, IQR, and 95% confidence intervals across seeds for:

- Survival and lifetime.
- Energy and resource collection.
- Reproduction rate.
- Field-use gain over the no-field control.
- Decorrelated-control penalty.
- Pre-intervention accuracy.
- Post-intervention accuracy.
- Adaptation steps after intervention.
- Transfer to unseen field strengths and patterns.
- Counterfactual flow information gain, alongside the frozen-policy transfer score.
The current action schema has no dedicated bounded probe action, so this pilot reports no active-probing rate.

## Required controls

Use at least ten seeds per condition for a headline result.
Use a separate intervention seed stream.
Keep world size, training budget, population size, and optimizer settings fixed.
Record the source commit, configuration, seed schedule, machine label, and cache declaration.

Do not use the direct-access condition as evidence of discovery.
Do not use a hidden-state probe as sole evidence of an internal model.
Representation must be paired with intervention and transfer behavior.

## Decision rules

A field is functionally used only when indirect access beats the decorrelated and no-field controls on the preregistered outcome.

A field relationship is functionally modeled only when the organism transfers across at least one held-out intervention and adapts faster than the decorrelated control.

A result that fails these rules remains engineering or preliminary evidence.

## Current limitation

The existing simulator exposes many world fields through the ordinary observation patch.
The field-masking controls must be enabled before the indirect-access condition is valid.
The existing PPO-POET campaign tests environment adaptation.
It does not test field discovery or consciousness.
