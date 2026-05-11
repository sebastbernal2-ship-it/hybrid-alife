# EC Reproducibility Checklist

This repo adopts the 5-dimension Evolutionary Computation reproducibility
checklist of López-Ibáñez et al. (arXiv:2602.07059). Tick every box before
publishing a result derived from this codebase.

The expectation is **descriptive, not prescriptive**: items marked `[ ]`
mean "not yet asserted by tests/CI"; they may be partially satisfied. A
release-quality run must populate this file with the actual values used.

---

## 1. Methodological clarity

- [x] Research question and hypothesis stated in `experiments/<run>.md`.
- [x] Algorithm pseudocode or reference implementation cited
      (`docs/scientific_validation.md`).
- [x] Population, selection, mutation, world step and termination
      criteria all defined by typed dataclasses in
      `src/hybrid_alife/types.py`.
- [x] Baseline algorithms named (objective-only, novelty-only, random,
      neutral-shadow). See `src/hybrid_alife/experiments/shadow.py`.

## 2. Experimental setup

- [x] Hyperparameters fully specified via YAML (`configs/*.yaml`).
- [x] Random seed fixed per run via `cfg.seed` and `jax.random.PRNGKey`.
- [x] Hardware noted alongside reports (CPU-only smoke run; GPU optional).
- [x] Code version: commit SHA captured in `outputs/<run>/config.yaml`.
- [x] World version: WorldConfig dump is part of the run artefact.
- [ ] ≥ 10 seeds per headline condition (current default: 3 for pilot; bump
      via `--seeds` in the ablation suite for release).

## 3. Results reporting

- [x] Median and IQR over seeds (`metrics.transfer.median_iqr`).
- [x] Effect size on every headline comparison
      (`metrics.transfer.cliffs_delta`).
- [x] Bootstrap CI (`metrics.transfer.bootstrap_ci`).
- [x] Bedau activity vs neutral-shadow control
      (`metrics.bedau.adaptive_activity`).
- [x] QD-score / coverage / archive entropy
      (`metrics.qd.qd_summary`).
- [x] Topsim + posdis + bosdis for any communication claim
      (`metrics.communication.comm_summary`).
- [x] Effective lineage count (Hill 1D) for diversity claims
      (`metrics.bedau.lineage_hill1d`).
- [x] Channel-shuffle / channel-zero ablations available
      (`metrics.communication.shuffle_channel|zero_channel`).
- [x] Uniform-field counterfactual ablation
      (`configs/ablation_uniform_field.yaml`).
- [x] Limitations + anti-overclaim language in reports
      (`scripts/generate_report.py`).

## 4. Artifact evaluation

- [x] `pip install -e .` reproduces the environment.
- [x] `pytest -q` exercises ≥80 unit / integration tests.
- [x] `scripts/run_ablation_matrix.py` runs the suite end-to-end on CPU
      under a short budget; outputs `outputs/<run>/report.md`.
- [x] CI runs the test suite on every push (`.github/workflows/ci.yml`).
- [ ] Notebooks pinned to specific commit (placeholder — add when
      `notebooks/` is fleshed out).

## 5. Paper / report metadata

- [x] Pre-registration template at `docs/preregistration_template.md`.
- [x] Run name, config, seed, commit SHA all in
      `outputs/<run>/config.yaml`.
- [x] Limitations / anti-overclaim section is mandatory in
      `scripts/generate_report.py`.
- [ ] DOI / archival snapshot — add at release time.

---

## How to fill this checklist for a release

1. Copy this file to `outputs/<run>/EC_REPRODUCIBILITY_CHECKLIST.md`.
2. Replace `[ ]` with `[x]` only where actually asserted by your run.
3. For ≥ 10 seeds, run `scripts/run_ablation_matrix.py --seeds 10`.
4. Commit the populated checklist alongside the report.
