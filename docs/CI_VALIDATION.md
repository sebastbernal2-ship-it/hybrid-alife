# CI Validation & Recovery

This document describes the CI surface for `hybrid-alife`, what it validates,
and how to recover quickly when something goes wrong.

## What CI runs

`/.github/workflows/ci.yml` runs on every push to `main` and on pull requests
targeting `main`. The job:

1. Checks out the repo.
2. Sets up Python on the matrix (`3.11`, `3.12`).
3. Installs the package and dev extras: `pip install -e ".[dev]"`.
4. Runs `ruff check src tests` (currently non-blocking via `|| true`).
5. Runs `pytest -q` — the smoke / unit test suite under `tests/`.

CI is intentionally lightweight: it does **not** run long JAX experiments,
multi-step ablations, or any of the `scripts/run_*.py` entry points end-to-end.
Those are exercised via dry-run / import smoke tests instead.

## Local reproduction

To reproduce CI locally:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
ruff check src tests || true
pytest -q
```

The full smoke suite is expected to finish in under a minute on a developer
laptop (no GPU, no long simulations).

## Critical commands validated by tests

The following entry points are covered by import / dry-run / parser tests so
that breakage is caught in CI without paying the cost of a full run:

| Command                         | Test file                          | What is checked                                  |
| ------------------------------- | ---------------------------------- | ------------------------------------------------ |
| `scripts/run_sim.py`            | `tests/test_smoke.py`              | Module imports; basic simulation entry point.    |
| `scripts/run_comm_benchmark.py` | `tests/test_comm_benchmark.py`     | Import + benchmark wiring.                       |
| `scripts/run_transfer_matrix.py`| `tests/test_transfer_scaling.py`   | Import + transfer-matrix shapes.                 |
| `scripts/run_compute_scaling.py`| `tests/test_transfer_scaling.py`   | Import + compute-scaling wiring.                 |
| `scripts/run_ablation_matrix.py`| `tests/test_cli_dry_run.py`        | `--help` / dry-run exits cleanly.                |
| `scripts/generate_report.py`    | `tests/test_cli_dry_run.py`        | `--help` exits cleanly.                          |
| Config loading                  | `tests/test_config_load.py`        | YAML configs under `configs/` parse and look sane.|

## Recovery playbook

When CI fails, follow this triage order:

1. **Lint regression** — `ruff check src tests` (currently advisory). If you
   want it to gate merges, drop the `|| true` in `ci.yml` once the tree is clean.
2. **Import error** — usually a missing dependency in `pyproject.toml`
   `[project.optional-dependencies].dev` or a circular import. Reproduce with
   `python -c "import hybrid_alife"` (or the failing module).
3. **Pytest failure** — run `pytest -q tests/<file>::<test>` locally with
   `-x --tb=short` to see the first failure.
4. **CLI breakage** — try the failing script with `--help`; many regressions
   show up as argparse / import errors at this stage.
5. **Config drift** — run `pytest tests/test_config_load.py` to confirm all
   YAML configs still load and validate.

If a recent merge introduced the regression, prefer reverting the offending
commit on `main` and re-fixing on a branch over leaving `main` red.

## Adding a new critical command

When introducing a new top-level `scripts/run_*.py` entry point:

1. Make sure `python scripts/run_<name>.py --help` exits 0 without importing
   heavy optional dependencies.
2. Add a one-line dry-run test to `tests/test_cli_dry_run.py`.
3. If the script reads a config under `configs/`, ensure the config is
   discovered by `tests/test_config_load.py`.

## Non-goals for CI

- Full end-to-end training / evolutionary runs.
- GPU / TPU acceleration paths.
- Long Monte-Carlo statistical validation (these live in
  `experiments/` and are run manually or out-of-band).

The intent is: **fast, deterministic, broad coverage of import & CLI
surface area**, with deeper science validation gated behind opt-in
experiment scripts.
