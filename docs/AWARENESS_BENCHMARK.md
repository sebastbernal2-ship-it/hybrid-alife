# Selective causal inquiry benchmark

This benchmark is a calibration layer for the field-discovery program.
It is not a hybrid-simulator result and it cannot establish consciousness.

## Research target

The target behavior is selective causal inquiry.
An agent should probe when information is useful, use the discovered rule, adapt when the rule reverses, transfer across surface changes, and avoid inventing structure when no stable rule exists.

## World ladder

- `solvable`: a probe reveals a stable hidden rule.
- `surface_shift`: the same hidden rule appears with a different nuisance surface.
- `reversal`: the hidden rule changes during the episode.
- `deceptive`: a visible cue points to the wrong rule.
- `impossible`: outcomes are independent of any hidden rule.
- `non_identifiable`: a generic probe is insufficient; a targeted probe is required.

The actions are exploit A, exploit B, probe, abstain, and two targeted probes.
All transitions use fixed seeds and the benchmark has no third-party dependency.

## Validity gate

Run the gate before using a world family in an expensive campaign.
It requires that:

- an oracle beats a no-signal policy in solvable worlds;
- impossible worlds expose no informative probe;
- surface shifts preserve the latent rule;
- reversal changes the rule;
- non-identifiable worlds require targeted probing.

The gate checks environment mechanics.
It does not show that a learned hybrid-alife controller can use them.

## Command

```bash
PYTHONPATH=src python scripts/run_awareness_benchmark.py \
  --seeds 0 1 2 \
  --output /tmp/awareness-benchmark.json
```

The artifact is marked `calibration_only`.
The field-discovery simulator now has an optional probe gate and records probe rate, returned signal magnitude, and probe-outcome information in its rollout artifacts.
Field-discovery configs also enable flow advection of resources and a bounded flow-alignment reward so hidden flow can affect the gathering task; the no-field control disables both couplings.
The field-discovery evaluator now scores held-out directional choices on a separate procedural seed and reports probe-conditioned alignment without exposing hidden flow to the controller.
The simulator metrics remain diagnostic until longer held-out runs, cue-scrambling controls, and preregistered uncertainty estimates are completed.
