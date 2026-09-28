# PPO-POET campaign archive

This directory archives the validated 20-cell PPO-POET campaign.

## Campaign design

- Tracks: `isolated` and `avida_enabled`.
- Seeds: `0` through `9` per track.
- Generations: `200` per cell.
- World size: `8` by `8`.
- Population size: `2` policies and `2` environments.
- Rollout horizon: `8` steps.
- Transfer horizon: `8` steps.
- Avida comparator horizon: `4` steps for this 200-generation campaign.
- Device: CPU.

The campaign manifest is `manifest.json`.
The source configuration is `configs/poet_campaign.yaml`.

## Results

The final transfer score mean was `17.9232` for `isolated`.
The final transfer score mean was `18.0641` for `avida_enabled`.
The paired mean difference was `+0.1409` for `avida_enabled - isolated`.
The Avida-enabled track was higher for `7` of `10` paired seeds.
The Cliff's delta was `0.08`.
The exact paired sign-permutation p-value was `0.8457`.

These results are descriptive.
They do not establish a transfer advantage for Avida coupling.

## Performance profile

The pre-JIT 200-generation profile took `381.378` seconds.
The post-JIT 200-generation profile took `294.866` seconds.
The observed reduction was about `22.7%`.

The profile files are `profile_pre_jit_200.json` and `profile_post_jit_200.json`.

## Provenance

The raw campaign files were copied from a local campaign staging directory after validation.
The original campaign manifest did not record its source Git commit.
The campaign used equal rollout and transfer horizons, so the later transfer-horizon fix does not change these campaign scores.
The later Avida metadata fix adds reporting for the comparator cap but does not change the comparator calculation.

`SHA256SUMS` records hashes for every archived file.
The PNG in `analysis/transfer_by_track.png` is the aggregate comparison figure.
The per-cell directories contain metrics, checkpoints, configurations, summaries, transfer matrices, and environment archives.

## Interpretation boundary

The PPO-POET module implements a bounded paired policy and environment population loop.
This archive does not support claims of open-ended evolution, language, or broad generalization.
Fixed-policy transfer and Avida comparison results must be read with `docs/poet_transfer.md` and `docs/SCIENTIFIC_INTERPRETATION_GUIDE.md`.
