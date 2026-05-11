# Campaign Acceleration & Caching

Practical knobs for shortening long experiment campaigns without rewriting the
runner. All features are opt-in and degrade gracefully if cache state is
missing or stale.

## TL;DR — workflow

```bash
# 1. Enable JAX persistent compilation cache for the whole shell session.
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1

# 2. First run: populate caches.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml

# 3. Re-run safely: completed cells are skipped automatically (resume mode).
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml --resume

# 4. Force a full re-execution (ignore prior results).
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml --force
```

## Cache layers

| Layer | What it caches | Where it lives | Speedup |
|---|---|---|---|
| JAX persistent compile cache | XLA-compiled HLO | `$JAX_COMPILATION_CACHE_DIR` | 2–10x on warm runs (compile dominates short cells) |
| Campaign run-skip cache | Completed cell outputs | `outputs/.../<run_name>/metrics.json` | Skips repeat work entirely on `--resume` |
| Per-cell config hash | Detects config drift | `<cell>/cell_config.yaml` (hash in manifest) | Invalidates stale skips automatically |

## `--resume` semantics

A cell is considered **complete** (and skipped on `--resume`) when *all* of:

- Its resolved `cell_config.yaml` exists and its content hash matches the
  freshly planned one.
- A `metrics.json` (or `metrics.csv`) exists in its output dir and is non-empty.

If either condition fails, the cell runs normally. The manifest records
`status: "cached"` for skipped cells with `duration_s: 0.0` and the cached
config hash, so resumed campaigns are still fully audited.

## `--force` semantics

`--force` disables all skip logic — every cell is materialized and executed
even if a prior `metrics.json` exists. Use this after changing code that the
config hash cannot detect (e.g., a bugfix in `src/`).

## Config hashing

The launcher computes a deterministic SHA-256 over the resolved per-cell
config (after seed/run_name/output_dir/generations overrides are applied).
This hash is written to the manifest as `cells[i].config_hash` so you can:

- Diff manifests across runs to see which cells genuinely changed.
- Detect "I changed a seed, why didn't anything resume?" (the hash will differ).

## JAX compilation cache

Set these env vars *before* invoking the launcher. They are read once by JAX
on import and apply globally:

```bash
export JAX_COMPILATION_CACHE_DIR="$PWD/.jax_cache"
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
```

Notes:

- The cache directory is keyed by JAX version, XLA backend, and HLO graph. It
  is safe to share across runs with the *same* environment, but you should
  blow it away (`rm -rf .jax_cache`) after JAX or driver upgrades.
- For repeatable CI, point `JAX_COMPILATION_CACHE_DIR` at a path inside the
  repo workspace (as above) so cache survives between CI steps.

## Parallel workers (experimental)

`--workers N` runs up to `N` cells concurrently via `concurrent.futures`. Use
only on machines where the per-cell process actually has free CPU/GPU
headroom — most cells in this codebase saturate JAX threads, so the safe
default is `--workers 1`. The manifest records the effective worker count.

## Limitations

- `--resume` trusts `metrics.json` as proof of completion. If a cell crashed
  *after* writing partial metrics but *before* writing all expected
  artifacts, you must `rm` its output dir or use `--force`.
- Config hashing does not include `src/` code changes. Use `--force` after
  fixing a bug in evolution logic.
- The JAX cache is process-local on first warm-up — parallel workers will
  each independently compile the first time they see a new HLO.
