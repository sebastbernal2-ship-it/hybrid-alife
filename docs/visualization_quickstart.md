# Visualization Quickstart

`scripts/visualize_run.py` turns whatever artifacts a run dropped on disk
into a static PNG gallery + `index.html` you can open in a browser. It is
deliberately lightweight — pure `matplotlib` + `numpy`, no JAX import,
no long simulation runs.

## When to use it

- After any `scripts/run_sim.py` run that produced `metrics.jsonl`.
- After a campaign step that emitted `map_elites.npz` and/or
  `novelty_archive.npz`.
- To get a quick visual sanity check that complements the markdown report
  from `scripts/generate_report.py`.

It is robust to missing artifacts: anything absent or malformed is skipped
and noted in the gallery, so partial run directories still produce useful
output.

## Usage

```bash
# Default: writes PNGs + index.html into <run_dir>/viz/
python scripts/visualize_run.py outputs/runs/smoke200

# Custom output directory
python scripts/visualize_run.py outputs/runs/smoke200 --out-dir /tmp/viz

# Custom title for the HTML report
python scripts/visualize_run.py outputs/runs/smoke200 --title "Smoke200 — final"

# Higher-resolution PNGs
python scripts/visualize_run.py outputs/runs/smoke200 --dpi 160
```

Then open `outputs/runs/smoke200/viz/index.html` in a browser.

## What it produces

Each panel is a separate PNG so individual plots can be embedded
elsewhere (slides, PR comments, README).

| Artifact required        | Output file               | Content                                |
| ------------------------ | ------------------------- | -------------------------------------- |
| `metrics.jsonl`          | `timeseries_qd.png`       | QD score, MAP-Elites coverage, archive size |
| `metrics.jsonl`          | `timeseries_novelty.png`  | Novelty archive size / mean / threshold |
| `metrics.jsonl`          | `timeseries_lineage.png`  | Lineage depth & Hill numbers           |
| `metrics.jsonl`          | `timeseries_enrichment.png` | Field stats (enrichment, concentration, metabolite) |
| `metrics.jsonl`          | `timeseries_communication.png` | action entropy, message energy, comm usage |
| `metrics.jsonl`          | `timeseries_population.png` | Alive fractions, merit, tasks solved |
| `map_elites.npz`         | `map_elites.png`          | Heatmap of `fitness` masked by `filled` |
| `novelty_archive.npz`    | `novelty_archive.png`     | 2-D scatter of descriptors (or histogram for 1-D) |
| _(any of the above)_     | `index.html`              | Gallery linking every panel produced   |

A metric group is silently skipped if none of its keys appear in
`metrics.jsonl`, so older runs with fewer columns still produce a tidy
report rather than a wall of empty plots.

## Pairs well with

- `scripts/generate_report.py` — the markdown headline-metrics report.
  Run both for a complete picture: numbers and pictures.
- `src/hybrid_alife/viz/plots.py` — the in-repo plotting primitives.
  `visualize_run.py` intentionally re-implements its own metric grouping
  (rather than calling those primitives) so it can run against partial
  artifacts without requiring a checkpoint decode.

## Limitations

- No checkpoint decoding: world fields and agent positions live in
  `checkpoint_final.pkl` and require importing JAX-y modules. For those
  use `hybrid_alife.viz.plots.plot_world_fields` /
  `plot_agent_positions` directly, or run `generate_report.py`.
- Static only: no interactive HTML, no animations. The HTML is just a
  gallery of PNGs.
- The MAP-Elites heatmap assumes a 2-D archive (the campaign's default).
  Higher-dimensional archives are skipped with a note.
- The novelty archive plot accepts arrays under the keys `behaviors`,
  `descriptors`, `points`, or `archive`. Unknown layouts are skipped.
