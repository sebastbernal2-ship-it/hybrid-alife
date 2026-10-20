# Independent replication on HiperGator

Status: **passed**, calibration only.

## Provenance

- commit: `6e3497ad2dcc49b542e01e114daab699a630836e` (`main`)
- host: `login7.ufhpc`
- nproc: `64`
- started_utc: `2026-10-20T05:44:01Z`
- cache_cleared: `true`
- python: `Python 3.12.5`
- jax: `0.12.0`

The run used a fresh clone of the canonical repository with `.jax_cache`
removed before starting.

Command:

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py \
  --seeds 0 1 2 3 4 \
  --output outputs/independent_replication_v1/summary.json
```

Artifact: `docs/results/gravity_study_v1/summary_hipergator.json`

## Agreement with the canonical artifact

| Arm | Local | HiperGator |
| --- | --- | --- |
| Calibration end to end | 5/5 seeds, worst +9.9355, mean +10.85 | 5/5 seeds, worst +9.9355, mean +10.85 |
| Simulator internal estimate | 5/5 seeds, worst +10.0561, mean +10.92 | 5/5 seeds, worst +10.0561, mean +10.92 |
| Revelations | none +11.00, true +12.00, false +10.00 | none +11.00, true +12.00, false +10.00 |

Per-seed worst-direction and mean values agree exactly across both arms, with a
maximum absolute difference of `0.000e+00`. The per-direction dictionaries differ
only at floating-point precision below the reported four decimals, which is
expected across machines.

## What this establishes

The result is not an artifact of one machine. The same seed schedule on a
different host, with a cleared cache and a fresh checkout, reproduces the gate
outcomes and the aggregate numbers.

## Limits

- Same JAX version (`0.12.0`) and same Python minor version (`3.12`) on both machines.
- Same operator and same repository. This is a computational replication, not a
  fully human-independent one.
