# Active-probe diagnostic smoke

This archive records one simulator-backed diagnostic smoke run after adding the optional active-probe channel.

- Source behavior: field-discovery configs enable the probe gate.
- Conditions: direct, indirect, decorrelated, no-field, and intervention.
- Seed: `0`.
- Training generations: `1`.
- Evaluation steps: `4`.

The direct, indirect, decorrelated, and intervention conditions returned nonzero probe signals.
The no-field control returned zero probe signal and zero probe-outcome information.

This is a diagnostic smoke artifact, not evidence of learned active probing.
The controllers were not evaluated on held-out procedural worlds, and one seed cannot support a scientific claim.
