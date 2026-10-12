# Field-discovery pilot

This pilot trains each condition with the existing hybrid-alife evolutionary runner, then evaluates a frozen controller through direct, indirect, spatially decorrelated, no-field, and intervention conditions.
It uses `run_experiment`, `initialize_sim`, and `step_sim` from the experiment runner.
It does not change PPO-POET criteria.

Run a small pilot with:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_field_discovery.py --seed 7 --steps 16 --training-generations 2 --output outputs/field-discovery-pilot.json
```

The runner saves one compact JSON artifact with schema version 4.
It records each condition's configuration, survival, energy, resource, and reproduction metrics.
It records scores before and after intervention, plus a held-out frozen-policy evaluation at 1.5 times the training flow strength.
The intervention condition reverses the flow vectors halfway through evaluation.
The decorrelated condition translates observed flow by half the toroidal world while retaining the original movement coupling and field-value distribution.
The no-field condition hides flow-derived observations and disables only flow-driven movement coupling.
Counterfactual flow information gain is the mutual information between flow magnitude and enrichment minus the same mutual information after translating flow by half the world.
The translated field retains its values and breaks their spatial relationship with enrichment.
This is an observational diagnostic, not an active-probing measure or proof of internal knowledge.

The score is survival fraction multiplied by mean energy among living embodied agents.
It is an operational pilot measure, not a validated discovery metric.
Resource is the final total in the world, not a cumulative collection count.
Reproduction is the number of births recorded during the run.

Aggregate artifact files or directories with:

```bash
PYTHONPATH=src python scripts/analyze_field_discovery.py outputs/field-discovery-pilot.json
```

This small pilot is an integration check, not evidence for a functional model or consciousness.
The controller action schema has no dedicated bounded probe action, so the pilot does not report an active-probing metric.
Do not interpret direct-access performance as discovery.
Research conclusions require more seeds, longer training, transfer tests, and controls.
