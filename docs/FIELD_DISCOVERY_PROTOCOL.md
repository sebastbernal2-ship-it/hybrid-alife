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

### Shuffled control

Preserve the field value distribution.
Break its spatial or temporal relationship with outcomes.

This condition tests whether performance depends on the field relationship rather than field statistics alone.

### Intervention

Reverse the field direction, change its strength, or shift its spatial pattern after training.
Evaluate frozen policies before adaptation and after a fixed adaptation budget.

This condition tests whether the organism learned a reusable relationship.

## Outcome levels

- **No use:** field access does not improve survival, energy, reproduction, or resource collection.
- **Incidental use:** performance improves in the training distribution but fails after intervention.
- **Implicit use:** the organism exploits the field without direct field observations.
- **Functional model:** the organism predicts consequences, transfers across field changes, and adapts faster than shuffled controls.
- **Active inquiry:** the organism selects actions that reduce uncertainty or reveal field consequences.

These levels describe behavior.
They do not establish subjective experience.

## Metrics

Report median, IQR, and 95% confidence intervals across seeds for:

- Survival and lifetime.
- Energy and resource collection.
- Reproduction rate.
- Field-use gain over the no-field control.
- Shuffled-control penalty.
- Pre-intervention accuracy.
- Post-intervention accuracy.
- Adaptation steps after intervention.
- Transfer to unseen field strengths and patterns.
- Active information-seeking rate when probing actions exist.

## Required controls

Use at least ten seeds per condition for a headline result.
Use a separate intervention seed stream.
Keep world size, training budget, population size, and optimizer settings fixed.
Record the source commit, configuration, seed schedule, machine label, and cache declaration.

Do not use the direct-access condition as evidence of discovery.
Do not use a hidden-state probe as sole evidence of an internal model.
Representation must be paired with intervention and transfer behavior.

## Decision rules

A field is functionally used only when indirect access beats the shuffled control and no-field control on the preregistered outcome.

A field relationship is functionally modeled only when the organism transfers across at least one held-out intervention and adapts faster than the shuffled control.

A result that fails these rules remains engineering or preliminary evidence.

## Current limitation

The existing simulator exposes many world fields through the ordinary observation patch.
The field-masking controls must be enabled before the indirect-access condition is valid.
The existing PPO-POET campaign tests environment adaptation.
It does not test field discovery or consciousness.
