# Pre-registration: plain channel against uniform, 40 seeds per arm

Committed before any seed 15-39 was run. Protocol, flags and gates are those of `../README.md`
(`--law-profile uniform` against `--law-profile lbm-geometry --no-obstacle`).

- **Confirmatory (one test):** paired two-sided Wilcoxon signed-rank on `min_delta`
  (plain channel minus uniform) over the **new seeds 15-39 only**, alpha 0.05. Seeds 0-14
  produced the hypothesis, so they are not part of this test.
- **Descriptive:** the same test over seeds 0-39; mean paired change with a seed-bootstrap 95%
  interval (10000 resamples); pass counts with Fisher exact and exact McNemar; the evolved and
  initial parts of the margin separately (mean over the four diagonals).
- **Reproduction gate:** seeds 0 and 1 of both arms are rerun on the current code and must
  match the committed `min_delta` within 1e-5. If not, seeds 0-14 are rerun as well and the
  mismatch is reported.
