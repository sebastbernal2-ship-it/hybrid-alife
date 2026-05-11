# Lineage Growth (Tiny-CPU Births Demo)

`configs/lineage_growth.yaml` is a minimal CPU-friendly experiment that
demonstrates real embodied reproduction: births occur, parent/child IDs are
set, and `max_lineage_depth_embodied` is strictly positive.

## Why it works

Two knobs added to `EmbodiedConfig` make births dependable on small worlds:

* `initial_alive_fraction` — fraction of slots that start alive. Default
  `1.0` (legacy behavior). Setting it below `1.0` leaves dead slots open so
  reproduction has somewhere to place children right away.
* `repro_gate_threshold` — sigmoid threshold the reproduce gate must exceed.
  Default `0.5` (legacy). Lowering it (e.g. `0.3`) makes untrained controllers
  reliably trigger births in tiny runs.

The runner now also tracks `embodied_births_per_generation` and
`embodied_deaths_per_generation` in `metrics.jsonl` so post-hoc analysis can
verify lineage activity without re-running.

## Run

```
python -c "from hybrid_alife.experiments.runner import load_config, run_experiment; run_experiment(load_config('configs/lineage_growth.yaml'))"
```

Expected outcome on a fresh seed:

* `max_lineage_depth_embodied >= 1`
* Total `embodied_births_per_generation` across the run `> 0`

## Test

```
pytest tests/test_lineage_growth.py -q
```
