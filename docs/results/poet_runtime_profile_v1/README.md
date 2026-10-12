# PPO-POET runtime profile v1

This record bounds the local CPU timing path before another preregistered campaign.

- Configuration: `configs/poet_campaign.yaml`.
- Tracks: `poet` and `static`.
- Seeds: one seed, `0`.
- Machine: local CPU.
- Archive selection limit: `16`.
- Causal validity limit: `16`.

## Measurements

- Two generations per track completed in `37.57` seconds.
- Twenty generations per track completed in `2:23.34`.
- The 200-generation profiled rerun completed one first cell in about `21` minutes before it was stopped.

The full 40-cell campaign was therefore still projected to take several hours.
The dominant code path is serial frozen-policy rollout evaluation in transfer matrices, with Python scalar conversions inside each rollout step.

These timings are engineering evidence only.
No PPO-POET scientific result is attached to this profile.
