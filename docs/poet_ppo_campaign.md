# Paired PPO-POET campaign

The standalone `hybrid_alife.poet` module implements paired policy and environment populations.

The policy is an actor-value multilayer perceptron trained with clipped PPO.

Each environment genome stores bounded scalar parameters and a fixed-shape terrain map.

Each generation evaluates every frozen policy on every environment.

The least useful environment is replaced by a bounded mutation of the best environment.

The least useful policy is replaced by a mutation of the best policy after PPO updates.

## Smoke run

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_poet.py \
  --config configs/poet_smoke.yaml \
  --seeds 0 1 \
  --generations 2 \
  --out-dir /tmp/hybrid-alife-poet-smoke
```

The default command runs both `isolated` and `avida_enabled` tracks.

The Avida-enabled track runs the existing Avida VM as a bounded comparison sidecar.
Its `summary.json` records the actual `avida_comparator_steps` cap used for that metric.

## Preregistered campaign

`configs/poet_campaign_manifest.json` is the campaign manifest.

It contains two tracks, ten seeds per track, and 200 generations per cell.

The manifest has 20 cells in total.

Run a campaign with `configs/poet_campaign.yaml` and the same `scripts/run_poet.py` entrypoint.

## Outputs

Each cell writes `metrics.jsonl`, `transfer_matrix.json`, `environment_archive.json`, `summary.json`, and `poet_checkpoint.pkl`.

Each cell also writes `figures/poet_progress.png`.

Aggregate completed cells with:

```bash
venv/bin/python scripts/analyze_poet_campaign.py /tmp/hybrid-alife-poet-smoke
```

The analysis writes `analysis/statistical_summary.json` and `analysis/transfer_by_track.png`.

The summary reports mean, median, standard deviation, IQR, and Cliff's delta when both tracks exist.

## Profiling rule

Record a full-run profile before adopting optimization changes:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/profile_full_run.py \
  --config configs/poet_smoke.yaml \
  --generations 2 \
  --out /tmp/hybrid-alife-full-profile.json
```

The first 200-generation profile recorded JAX dispatch overhead in world and Avida steps.

JIT boundaries were adopted only after that profile and the post-change profile records the adoption.

The legacy runner remains unchanged by this module.
