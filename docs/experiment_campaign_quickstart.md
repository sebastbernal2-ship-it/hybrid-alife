# Experiment campaign quickstart

`scripts/run_quick_campaign.py` orchestrates a small ablation matrix
(configs × seeds) over the existing single-run entrypoint
(`scripts/run_sim.py`). It avoids the chore of stitching together
`run_sim.py --config ...` invocations by hand when you want to fan out a
sanity-check sweep before committing to a long study.

## What it does

For each cell of the matrix the launcher:

1. Loads the cell's source config (one of the existing `configs/*.yaml`).
2. Applies overrides — `seed`, `run_name`, `output_dir`, and optionally
   `evolution.generations`.
3. Writes a fully-resolved per-cell config into the cell's output directory
   (`<out>/<run_name>/cell_config.yaml`).
4. Invokes `python scripts/run_sim.py --config <cell_config>` as a
   subprocess.

A `manifest.json` and `manifest.md` are written to the campaign output
directory regardless of whether cells executed — so a dry-run produces a
complete plan that can be turned into a real run by re-invoking without
`--dry-run`.

## Campaign config

See `configs/campaigns/quick_20min.yaml` for the canonical example.

```yaml
name: quick_20min
description: >
  Baseline + two structural ablations, two seeds each.
base_output_dir: outputs/quick_campaign
configs:
  - configs/smoke200.yaml
  - configs/ablation_no_comms.yaml
  - configs/ablation_no_memory.yaml
seeds: [0, 1]
generations: 20   # optional; applied to every cell
```

Required keys: `configs`. Everything else has defaults.

## Usage

```bash
# Plan only — no subprocesses launched. Writes manifest + per-cell configs.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --dry-run

# Execute the full matrix.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml

# Cap wall time by limiting the number of cells executed. Remaining cells
# are listed in the manifest as `skipped` so you can resume later by hand.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --max-runs 3

# Override seeds and generations from the CLI (takes precedence over YAML).
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --seeds 0 1 2 \
    --generations 5

# Custom output directory.
python scripts/run_quick_campaign.py \
    --campaign configs/campaigns/quick_20min.yaml \
    --out-dir outputs/exp_2026_05_11
```

## Flags

| Flag | Description |
|------|-------------|
| `--campaign PATH` | Campaign YAML (default `configs/campaigns/quick_20min.yaml`). |
| `--dry-run` | Plan only. Write manifest + per-cell configs; launch no subprocesses. |
| `--max-runs N` | Hard cap on executed cells. Remaining cells are marked `skipped`. |
| `--seeds N [N ...]` | Override the YAML seed list. |
| `--generations N` | Override `evolution.generations` for every cell. |
| `--out-dir PATH` | Override the campaign's `base_output_dir`. |
| `--run-sim PATH` | Alternate single-run entrypoint (defaults to `scripts/run_sim.py`). |
| `--python PATH` | Python interpreter for subprocess invocations. |

## Manifest

`<out>/manifest.json` records:

- Campaign metadata (name, description, source path).
- Invocation (argv, overrides, dry-run flag).
- Environment (platform, Python version, working dir, git SHA, timestamp).
- Per-cell records: source config, seed, run name, output dir, the resolved
  per-cell config path, exact subprocess command, status (`planned`,
  `dry-run`, `ok`, `failed`, `error`, `skipped`), return code, duration,
  and start/finish timestamps.

`<out>/manifest.md` is a human-readable rendering of the same data with a
cell table and the list of commands.

## Exit code

Non-zero if any cell ended with status `failed` or `error`. `skipped` and
`dry-run` cells do not affect the exit code.

## Limitations

- Cells run sequentially. No parallelism / scheduler integration.
- Subprocess stdout/stderr is inherited, not captured into the manifest.
- The override surface is limited to `seed`, `run_name`, `output_dir`, and
  `evolution.generations`. Arbitrary nested overrides (e.g. world size)
  are not yet exposed — for those, edit the underlying config or add a new
  one to `configs/`.
- The launcher assumes `scripts/run_sim.py` accepts `--config PATH`. If
  you point `--run-sim` at a different entrypoint it must accept the same
  flag.
