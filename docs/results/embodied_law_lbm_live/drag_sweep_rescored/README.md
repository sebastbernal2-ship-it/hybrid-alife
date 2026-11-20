# Is the drag-sweep trend baseline inflation? (pre-registration, results to follow)

Status: **bounded pilot, 15 seeds. The design and tests below were pre-registered before any
data were produced (commit `ca3a55e`); the run followed them with no deviation. The
pre-registered primary test did NOT find that the trend vanishes: under the law target the
per-seed slope is -0.060 per unit drag (p = 0.005). Under the own-wake-removed target the slope
shrinks by about 60% and is no longer distinguishable from zero (p = 0.17), and genomes trained
at higher drag are not worse when evaluated uncoupled. So the sweep trend is partly baseline
inflation by the agent's own wake and partly a real behavioural change that the study cannot
cleanly separate. Not a claim.**

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

## Results (run after the pre-registration)

Reproduction gate: the `official` min_delta per seed and drag equals the committed per-seed
numbers exactly (max absolute difference 0.0 at all seven drags), so the retrained genomes
(drag 0.5, 0.75, 1.0, 1.25, 1.5) and the copied mechanism-study caches (drag 0, 2.0) are the
genomes the sweep scored. Files: `eval_per_seed/` (every seed), `analysis.json`,
`analysis.txt` (full script output).

### Pre-specified slope tests (per-seed slope of min_delta against drag, seven drags)

| Target | Mean slope [95% CI] | Seeds negative | One-sided Wilcoxon p |
|---|---|---|---|
| **law (primary)** | **-0.060 [-0.097, -0.020]** | 12 / 15 | **0.0051** |
| official (reproduction) | -0.068 [-0.103, -0.033] | 13 / 15 | 0.0017 |
| noself (secondary) | -0.027 [-0.069, +0.013] | 8 / 15 | 0.165 |
| selfonly (diagnostic) | -0.093 [-0.126, -0.055] | 14 / 15 | 0.0006 |
| transfer: D-trained genomes evaluated uncoupled (secondary) | -0.010 [-0.046, +0.025] | 8 / 15 | 0.281 |

The `official` row reproduces the sweep's -0.068 / 13 of 15 / p = 0.0017 exactly. Pre-registered
reading of the primary test: the trend vanishes only if p > 0.05 and the interval includes zero;
the law-target slope fails both, so **the trend does not vanish under the law target**.

### Per drag (mean change in min_delta against drag 0, Holm over the six drags)

| Drag | official | law | noself | transfer |
|---|---|---|---|---|
| 0.5 | -0.034 (Holm 0.49) | -0.017 (0.76) | -0.011 (1.0) | +0.003 (1.0) |
| 0.75 | -0.060 (0.17) | -0.038 (0.56) | -0.025 (1.0) | -0.008 (1.0) |
| 1.0 | -0.124 (0.062) | -0.113 (0.17) | -0.083 (0.72) | -0.077 (0.81) |
| 1.25 | -0.080 (0.17) | -0.044 (0.56) | -0.015 (1.0) | -0.002 (1.0) |
| 1.5 | -0.114 (0.062) | -0.098 (0.091) | -0.051 (0.72) | -0.020 (1.0) |
| 2.0 | -0.134 (0.062) | -0.115 (0.11) | -0.054 (1.0) | -0.018 (1.0) |

No coupled drag is resolved after Holm correction under any target here, as in the sweep. The
seed-bootstrap 95% intervals of the law-target change exclude zero at drags 1.0, 1.5 and 2.0;
the noself and transfer intervals include zero at every drag (`analysis.txt`).

### Baseline inflation by drag (mean alignment, own fluid; official / law / noself)

| Drag | Initial population | Evolved population |
|---|---|---|
| 0 | 0.056 / 0.056 / 0.056 | 0.612 / 0.612 / 0.612 |
| 0.5 | 0.105 / 0.058 / 0.056 | 0.624 / 0.595 / 0.603 |
| 1.0 | 0.152 / 0.061 / 0.053 | 0.606 / 0.545 / 0.564 |
| 1.5 | 0.199 / 0.066 / 0.050 | 0.631 / 0.549 / 0.568 |
| 2.0 | 0.239 / 0.073 / 0.049 | 0.646 / 0.542 / 0.568 |

The own wake at the agent's cell grows roughly linearly with drag (0.018 at 0.5 to 0.072 at 2.0,
law magnitude 0.2) and points along the agent's own thrust at every drag (cosine 0.89 to 0.94).

Post-hoc, not pre-specified: the initial population's official alignment rises with drag on
all 15 seeds (mean slope +0.092 [+0.086, +0.099] per unit drag); removing the own wake removes
about 60% of the official min_delta slope (0.60, 95% interval 0.31 to 1.39).

## Reading

Supported:
- **The inflation grows smoothly with drag.** The initial policies' official alignment goes
  from 0.056 to 0.239 across drag 0 to 2.0, on every seed, and with the own wake removed it
  stays at 0.049 to 0.056. So the baseline inflation seen at drag 2.0 in `../mechanism/` is the
  top of a continuous curve, not a feature of one drag value.
- **Own-wake removal weakens the slope.** The slope falls from -0.068 to -0.027 and is no longer
  distinguishable from zero (p = 0.17), and no drag has a resolved change under `noself`.
- **Training at higher drag does not damage the policy.** D-trained genomes evaluated in the
  uncoupled fluid show no dose-response (slope -0.010, p = 0.28), the sweep-wide version of the
  E-to-B cross-evaluation in `../estimate_error/`.

Not supported or unresolved:
- **"The whole trend is baseline inflation."** The pre-registered law-target slope persists
  (-0.060, p = 0.005) and the interval for the noself slope (-0.069 to +0.013) still allows a
  slope as large as the official one (its lower bound is -0.069). The 60% figure has an interval reaching 1.39.
  With 15 seeds the data neither establish nor exclude a residual cost.
- **What the persisting law-target slope means.** The evolved population's alignment with the
  law falls with drag (0.612 to 0.542) because the evolved policies track the real local flow,
  wakes included, as `../mechanism/` found for drag 2.0 (its "H4"). That costs alignment with
  the law, not with the flow they are scored on. So the law target is not a clean measure of
  learning under coupling either, which is why `noself` is reported as a secondary target; the
  pre-registered primary stays the law target and is reported as pre-registered.

## Caveats

- 15 seeds, one carrier (14x14 periodic box with a mean-flow controller), an uncalibrated drag
  coefficient, monopole coupling only (the unphysical model).
- The `noself` target is a model-based estimate (shadow-fluid superposition). Its residual
  against the full wake grows with drag: mean 0.016 at drag 0.5 to 0.065 at 2.0 for trained
  genomes, with a maximum of 0.32 at drag 2.0 (larger than the 0.14 reported for B-trained
  genomes in `../mechanism/`, because these are E-trained genomes).
- Several targets and tests were examined; only the law-target slope was pre-specified as
  primary. Everything else is descriptive, and the post-hoc quantities are labelled.
- The Holm-corrected per-drag results are weak, as in the sweep itself.
- Genome caches are not committed (written to a scratch directory); retraining is
  deterministic and reproduces the committed per-seed numbers exactly (the gate above).
  Run with `python scripts/measure_sweep_rescored.py train --out DIR` then `eval`, then
  `python scripts/analyze_sweep_rescored.py --eval DIR/eval --json analysis.json`.

## Next

Score both populations against a target the agents cannot influence at all, for example a
no-coupling copy of the same trajectory, which avoids the superposition estimate and the
behaviour effect on the law target. Run a physical (dipole) comparison on the same pipeline;
the dipole's own wake is small, so it should show no slope under any target. About 40 seeds
per arm would be needed to resolve a residual slope of the size the noself interval allows.
