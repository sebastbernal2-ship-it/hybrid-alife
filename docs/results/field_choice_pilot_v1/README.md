# Held-out directional choice pilot v1

This archive adds a held-out directional choice metric to the simulator field-discovery evaluation.
The frozen controller chooses movement actions in a new procedural world seeded with the training seed plus `1_000_003`.
The evaluator scores movement alignment with local causal flow after the rollout.
It does not provide hidden flow to the controller.

- Seeds: `0`, `1`, and `2`.
- Training: two generations per condition.
- Held-out evaluation: eight steps per condition.
- Conditions: direct, indirect, decorrelated, no-field, and intervention.

## Finding

Held-out choice accuracy was below chance in this short pilot:

- Direct: `0.392`.
- Indirect: `0.403`.
- Decorrelated: `0.403`.
- Intervention: `0.403`.

Held-out choice alignment was also negative in all causal-field conditions.
The no-field control produced no valid causal choice events, as expected.
Probe-conditioned alignment was near zero and did not exceed non-probe alignment reliably.

This is a useful diagnostic negative result.
The current agents do not transfer a useful directional choice rule to a new procedural surface after two training generations.
The metric is now available for longer bounded training and archive-control comparisons.

This is engineering evidence only.
It does not establish environmental model-use or consciousness.
