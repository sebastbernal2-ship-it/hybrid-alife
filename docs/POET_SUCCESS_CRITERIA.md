# PPO-POET success criteria

This file owns the preregistered success gate for the bounded PPO-POET campaign.

## Primary comparison

Compare the dynamic `poet` track with the `static` track.
The primary outcome is the mean frozen-policy raw-reward transfer to four held-out environments per seed.
Held-out environments come from `seed + heldout_seed_offset` and never enter policy updates, selection, mutation, or archive admission.

## Environment controls

A candidate environment is admitted only when its deterministic frozen-policy raw reward is at least `minimal_criterion`.
The archive keeps every admitted candidate and records its parent, birth generation, genome, and admission score.
The final archive transfer matrix evaluates every frozen policy on every archived stepping stone.

The default campaign uses `minimal_criterion: -10.0` as a declared raw-reward lower bound.
This value is a campaign parameter, not a calibrated claim about open-endedness.

## Replication metadata

Every cell uses artifact schema version `2` and records:

- `replication_id`
- `seed_offset`
- `operator`
- `machine_label`
- `cache_cleared`

Independent replications must use distinct seed offsets and declare a cleared cache.
The analyzer does not infer independence from a different label.

## Strong success gate

The analyzer reports `success` only when all conditions hold:

1. At least two replication IDs are present.
2. Each replication has at least 10 paired seeds.
3. Each replication has a positive held-out effect for `poet - static`.
4. Each replication has exact paired sign-permutation `p < 0.05`.
5. The pooled paired-effect bootstrap 95% interval is entirely above zero.
6. Replication seed offsets are distinct and every replication declares `cache_cleared: true`.

A result with fewer requirements is `insufficient_evidence`.
It can still be useful engineering or preliminary evidence.

## Commands

Run the two replications into one nested campaign directory so the analyzer can discover and pair every cell:

```bash
for replication in 1 2; do
  offset=$(( (replication - 1) * 10000 ))
  JAX_PLATFORMS=cpu venv/bin/python scripts/run_poet.py \
    --config configs/poet_campaign.yaml \
    --tracks poet static \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --replication-id replication-$replication \
    --seed-offset "$offset" \
    --operator NAME \
    --machine-label MACHINE \
    --cache-cleared \
    --generations 200 \
    --out-dir docs/results/poet_campaign_final_v3/replication-$replication
done

venv/bin/python scripts/analyze_poet_campaign.py docs/results/poet_campaign_final_v3 \
  --primary-track poet --baseline-track static
```

The analysis artifact is `analysis/statistical_summary.json`.
The analyzer rejects stale schema versions, missing replication metadata, and duplicate replication-track-seed cells instead of overwriting them.
The decision is descriptive evidence only until an independent operator or machine completes the second replication.
