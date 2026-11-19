# Is the drag-sweep trend baseline inflation? (pre-registration, results to follow)

Status: **PRE-REGISTERED, NOT YET RUN.** This section was committed before any data were
produced; the results sections are appended after the run.

Parents: `../drag_sweep/` (monopole min_delta falls gradually with drag, per-seed slope
-0.068 per unit drag, p = 0.0017), `../mechanism/` (at drag 2.0 the loss is mostly inflation of
the *initial-policy* baseline by each agent's own wake, not worse evolved policies).

## Question

Does the drag-sweep dose-response survive when both populations are scored against a target the
agents cannot influence? Or is the whole trend the same baseline inflation, growing with drag?

## Design (fixed before the run)

- Monopole coupling at drag in {0.5, 0.75, 1.0, 1.25, 1.5, 2.0} plus the uncoupled live fluid
  (drag 0). Seeds 0-14, the recorded flow-head protocol, the same carrier as the parents.
- For each seed and drag D: genomes trained at D (retrained deterministically; B and E genomes
  come from the earlier mechanism caches when present) and the shared initial genomes are
  rolled out on the four held-out diagonals in the fluid at D, with one shadow fluid per agent
  (as in `../mechanism/`) so each rollout can be scored against four targets: `official`
  (local live flow), `law` (imposed direction), `noself` (local flow minus the agent's own wake),
  `selfonly`. A second set of rollouts evaluates the D-trained genomes in the uncoupled fluid
  (drag 0), which removes every evaluation-time effect (`transfer`).
- min_delta per target = min over diagonals of (evolved - initial), both scored in the same
  fluid against the same target.
- Reproduction gate: the `official` min_delta per seed and drag must equal the committed
  per-seed numbers in `../drag_sweep/` and `../` (max abs difference below 1e-5). If not, the
  run is invalid and is reported as such.

## Pre-specified tests

- **Primary (one test):** per-seed least-squares slope of **law-target** min_delta against drag
  across the seven drags (0, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0), one-sided Wilcoxon signed-rank
  on the 15 slopes against zero (alternative: negative), alpha 0.05. "The trend vanishes" means
  p > 0.05 **and** the bootstrap 95% CI of the mean slope includes zero; with n = 15 a
  non-significant slope is absence of evidence, not evidence of absence, and is reported with
  its interval.
- **Secondary (descriptive, same slope test, no correction):** the `noself`-target slope and
  the `transfer` slope (D-trained genomes evaluated uncoupled). The `official` slope is the
  reproduction (expected -0.068, p = 0.0017).
- **Per drag:** paired Wilcoxon of each target's min_delta against drag 0, Holm over the six
  coupled drags per target, and seed-bootstrap 95% CIs (5000 resamples) of the paired mean change.
- **Baseline inflation curve (descriptive):** mean alignment of the initial and the evolved
  populations under each target, by drag.
