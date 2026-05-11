# Active MAP-Elites / QD

`configs/qd_active.yaml` is a tiny CPU-only configuration that exercises the
MAP-Elites pathway end-to-end and produces a strictly positive archive
coverage in a short run (~20s wall on a single core).

## What it produces

Per generation, the runner now writes a `kind: "generation"` record into
`metrics.jsonl` with:

- `map_elites_coverage` — fraction of grid cells filled
- `qd_score` — sum of shifted fitness over filled cells
  (Mouret & Clune 2015 convention; non-negative)
- `archive_entropy` — Shannon entropy (nats) of the normalised fitness
  distribution over filled cells (high = uniformly-good archive, low =
  mode collapse)
- `novelty_archive_size`

These metrics are emitted regardless of whether `map_elites_enabled` is
true, so downstream tooling can always read a consistent schema.

## Why a tiny config still gets coverage > 0

The embodied behaviour descriptor is an EMA over per-step
`(speed, eat)` magnitudes. With the default 0.95 EMA retention,
a short run of a few generations leaves almost every agent near the
origin and only the (0, 0) bin ever fills.

`initialize_embodied_population` now seeds `behavior_descriptor` from
`Uniform([0.05, 0.95]^2)`. The EMA then drags each agent toward its
realised signature, but the initial spread guarantees that several
bins start populated. The archive update reflects this immediately on
the first end-of-generation.

## Running

```bash
python -m hybrid_alife.experiments.runner configs/qd_active.yaml
# or via pytest, end-to-end:
pytest tests/test_qd_active.py
```

Expected output (representative):

```
gen=0 cov=0.438 qd=20.5  ent=1.69
gen=5 cov=0.438 qd=78.0  ent=1.42
```

Coverage is stable across generations because the descriptor smoothing
keeps each agent close to its starting bin; QD-score and archive
entropy then evolve under selection.
