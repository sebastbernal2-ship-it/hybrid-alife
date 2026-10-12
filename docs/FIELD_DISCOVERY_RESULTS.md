# Field-discovery pilot

This pilot runs the direct, indirect, shuffled, and intervention conditions in the hybrid-alife simulator.
It uses `initialize_sim` and `step_sim` from the experiment runner.
It does not change PPO-POET criteria.

Run a small pilot with:

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_field_discovery.py --seed 7 --steps 16 --output outputs/field-discovery-pilot.json
```

The runner saves one compact JSON artifact.
It records the seed, config file, condition controls, and final survival, energy, resource, and reproduction metrics.
It also records scores before and after the halfway point.
The intervention condition turns on configured drift at that point.
Other conditions use the same point to record comparable snapshots.

The score is survival fraction multiplied by mean energy among living embodied agents.
It is an operational pilot measure, not a validated discovery metric.
Resource is the final total in the world, not a cumulative collection count.
Reproduction is the number of births recorded during the run.

Aggregate artifact files or directories with:

```bash
PYTHONPATH=src python scripts/analyze_field_discovery.py outputs/field-discovery-pilot.json
```

This small pilot is an integration check, not evidence for a functional model or consciousness.
Do not interpret direct-access performance as discovery.
Research conclusions require more seeds, transfer tests, and controls.
