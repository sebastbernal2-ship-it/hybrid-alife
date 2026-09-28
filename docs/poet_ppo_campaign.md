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

The default command runs `isolated` and `avida_enabled` tracks.

The `avida_enabled` track runs the existing Avida VM as a reset-per-rollout comparison sidecar.
The `avida_persistent` track carries its Avida population across PPO training rollouts and generations.
Both Avida tracks remain bounded comparisons, not open-ended evolution.
Their `summary.json` files record the lifecycle and the actual `avida_comparator_steps` cap used for the comparator metric.

## Preregistered campaign

`configs/poet_campaign_manifest.json` is the campaign manifest.

It contains three tracks, ten seeds per track, and 200 generations per cell.

The manifest has 30 cells in total.
The original 20-cell design is rerun with corrected raw-reward transfer scores, and the persistent Avida track adds 10 comparison cells.

Run a campaign with `configs/poet_campaign.yaml` and the same `scripts/run_poet.py` entrypoint.

## Outputs

Each cell writes `metrics.jsonl`, `transfer_matrix.json`, `environment_archive.json`, `summary.json`, and `poet_checkpoint.pkl`.

Each cell also writes `figures/poet_progress.png`.

Aggregate completed cells with:

```bash
venv/bin/python scripts/analyze_poet_campaign.py /tmp/hybrid-alife-poet-smoke
```

The analysis writes `analysis/statistical_summary.json` and `analysis/transfer_by_track.png`.

The summary reports mean, median, standard deviation, IQR, Cliff's delta, paired mean differences, wins, and exact paired sign-permutation p-values for every track pair.
Each cell records the source Git commit and complete configuration in its JSON artifacts.

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
