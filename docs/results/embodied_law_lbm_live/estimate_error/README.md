# Is the monopole margin loss a noisier flow estimate?

Status: **bounded diagnostic, 15 seeds per condition. The hypothesis that monopole coupling
lowers the margin by making the policy's flow estimate noisier is not supported. The estimate
error is small in every condition and does not rank the conditions the way the margin does.
A cross-evaluation shows the monopole-2.0 loss is mostly an evaluation-time effect of that
fluid, not damage done during training. The mechanism was unidentified here; the follow-up on
`exp/lbm-live-mechanism` (`../mechanism/`) found it: each monopole agent's own wake inflates
the baseline the margin is measured against. Not a claim.**

Parent study: `../README.md` (monopole drag 2.0 lowers min_delta by 0.134, 13 of 15 seeds
worse, Wilcoxon p = 0.012; dipole 2.0 has no effect).

## Question

The flow-head policy acts on the unit vector of `previous_velocity - previous_thrust`, which
is the flow that advected the agent during the previous step. Does coupling make that
estimate a worse predictor of the flow the agent is scored against, and does that explain the
margin loss?

## Method

- Scripts: `scripts/measure_flow_estimate_error.py` (training, evaluation, cross-evaluation)
  and `scripts/analyze_flow_estimate_error.py` (statistics). Helper and hook:
  `flow_estimate_errors`, `angular_error_deg` and an optional `diagnostics` list on
  `evaluate_frozen_alignment` in `latent_law_embodied.py`. The hook only records; it never
  changes a rollout. Unit tests: `tests/test_flow_estimate_error.py`.
- Conditions A-F and the protocol are those of the parent study, 15 seeds each. Every
  (condition, seed) was retrained from scratch.
- **Reproduction check:** per-seed min_delta equals the parent study's recorded value for all
  90 runs, maximum absolute difference 0.0. The diagnostic therefore describes the same
  genomes the parent study scored.
- **What is measured.** In the frozen evaluation rollouts of the evolved genomes (4 held-out
  diagonals, 12 steps, 16 agents each), after step *t* the estimate is
  `previous_velocity - previous_thrust`. The policy acts on it at step *t + 1* and is scored
  against the flow it finds after that step, so the error pairs `estimate[t]` with
  `flow[t + 1]` at the agent's cell (angle in degrees; agent alive at both steps). Also
  recorded: the angle between the local flow and the imposed law, the relative flow
  magnitude deviation `|flow - law| / |law|`, and the fraction of agent-steps with error
  above 45 degrees.
- Per seed, each metric is averaged over agent-steps within a direction, then over the four
  directions. Statistics use one number per seed.

## Results: estimate error by condition

| Condition | Estimate error, mean (deg) | Steps > 45 deg | Flow vs law angle (deg) | Flow magnitude deviation | Median min_delta |
|---|---|---|---|---|---|
| A legacy uniform | 0.00 | 0.000 | 0.00 | 0.000 | 0.393 |
| B live, no coupling | 0.00 | 0.000 | 0.00 | 0.000 | 0.393 |
| C monopole 0.5 | 1.08 | 0.000 | 3.87 | 0.176 | 0.328 |
| D dipole 0.5 | 1.22 | 0.000 | 2.16 | 0.065 | 0.325 |
| E monopole 2.0 | 4.14 | 0.003 | 13.33 | 0.686 | 0.255 |
| F dipole 2.0 | 6.33 | 0.007 | 9.66 | 0.271 | 0.418 |

Against B, every coupled condition has a higher estimate error on 15 of 15 seeds (Wilcoxon
p = 6e-5 each), so coupling does degrade the estimate. But it is small (a few degrees), and
**the order does not match the margin**: F (dipole 2.0) has the largest estimate error, 6.3
degrees against 4.1 for E, yet F loses no margin (+0.001 against A) while E loses 0.134.
In A and B the estimate is exact, because a uniform law makes the advecting flow equal to
the law.

## Does the error predict the margin?

| Pool | Spearman rho (p) | rho after removing per-condition means (p) |
|---|---|---|
| all six conditions (n = 90) | -0.15 (0.17) | -0.22 (0.037) |
| coupled C-F (n = 60) | -0.12 (0.37) | -0.30 (0.021) |

Per condition (n = 15): C -0.07 (0.81), D -0.32 (0.24), E -0.05 (0.85), F -0.59 (0.022).

Across conditions there is no association. Within conditions the sign is negative (more
error, lower margin) and nominally significant in two pooled tests and in F alone. These are
eight uncorrected Spearman tests (two pooled, two pooled within-condition, four per condition);
the smallest p, 0.021, does not survive a Bonferroni factor of eight (0.17). In E, the condition the question is about, rho is -0.05.

## Mediation check for the E-versus-B gap

Pooled B and E (n = 30), OLS of min_delta on a condition indicator with and without the
mediator, seed-resampled bootstrap, 4000 draws. total = gap, indirect = total - direct.

| Condition | Mediator | Total gap | Direct | Indirect [95% CI] |
|---|---|---|---|---|
| E | estimate error | -0.134 | -0.102 | -0.032 [-0.43, +0.31] |
| E | flow vs law angle | -0.134 | -0.219 | +0.085 [-0.62, +0.47] |
| E | flow magnitude deviation | -0.134 | -0.215 | +0.081 [-0.34, +0.39] |
| F | estimate error | +0.001 | +0.237 | -0.236 [-0.43, -0.10] |

For E the point estimate puts about a quarter of the gap on the estimate error (0.032 of
0.134), but the interval spans roughly -0.4 to +0.3, and the two other mediators give an
indirect effect of the opposite sign. This check cannot support or rule out mediation at
n = 15. The F row shows suppression, not mediation: F has more estimate error than E and no
gap, so adjusting for error makes F look better.

## Cross-evaluation: training versus evaluation

The main results train and evaluate in the same fluid, so a loss could come from damaged
training or from a harder evaluation. Genomes trained in one condition were therefore
evaluated under another's fluid, each against the initial genomes evaluated in that same
fluid (the initial genomes are identical across conditions for a seed; checked).

| Train -> eval | Median min_delta | Mean estimate error (deg) | Against B -> B: mean min_delta diff (Wilcoxon p, seeds worse) |
|---|---|---|---|
| B -> B | 0.393 | 0.00 | reference |
| E -> E | 0.255 | 4.14 | -0.134 (0.013, 13/15) |
| B -> E | 0.359 | 4.02 | -0.093 (0.001, 13/15) |
| E -> B | 0.372 | 0.00 | -0.018 (0.64, 9/15) |
| F -> E | 0.322 | 3.99 | -0.084 (0.018, 11/15) |
| F -> F | 0.418 | 6.33 | +0.001 (0.98, 7/15) |
| B -> F | 0.371 | 6.18 | -0.016 (0.33, 10/15) |
| F -> B | 0.427 | 0.00 | +0.009 (0.72, 6/15) |
| E -> F | 0.375 | 5.83 | -0.033 (0.45, 10/15) |

- Genomes trained uncoupled and evaluated in the monopole-2.0 fluid (B -> E) lose 0.093, about
  70% of the E -> E loss. Genomes trained in that fluid and evaluated uncoupled (E -> B) lose
  almost nothing (-0.018, p = 0.64). F-trained genomes also lose 0.084 when evaluated in E.
- So the loss belongs to evaluating in the monopole-2.0 fluid, whoever trained the genomes.
  Training in that fluid does not appear to damage the policy.
- No cell evaluated in the dipole fluid (B -> F, E -> F) shows a clear loss, although the
  dipole fluid gives the largest estimate errors.

## Reading

Supported:
- Coupling raises the flow-estimate error on every seed, from about 1 degree (drag 0.5) to
  4 to 6 degrees (drag 2.0). It stays small: under 1% of agent-steps exceed 45 degrees.
- The estimate error does not explain the monopole margin loss: the dipole at the same drag
  has more of it and no loss, and within E the error does not track the margin.
- The monopole-2.0 loss is mainly an evaluation-time effect of that fluid (B -> E).

Not supported or unresolved (the first item was resolved afterwards by the mechanism study on
`exp/lbm-live-mechanism`: self-contamination of the baseline by the agent's own wake):
- Which feature of the monopole fluid lowers the measured alignment. The monopole fluid is
  the one with the largest flow perturbation (magnitude deviation 0.69, direction 13 degrees
  off the law). The score is the alignment of the agent's move against the actual local flow,
  so a perturbed scoring target is a candidate. It was not isolated here.
- Whether the within-condition negative trend is real; it is weak and uncorrected.
- Any mediation claim; the intervals are far too wide.

## Caveats

- Same-fluid results use the parent study's training and evaluation in the same fluid. The
  cross-evaluation changes only the evaluation fluid (the initial genomes are re-scored in it).
- n = 15 seeds per cell; every p-value here is uncorrected unless stated.
- The mediators and the outcome are measured on the same evaluation rollouts, so they share
  noise; the mediation estimates are descriptive.
- The estimate is compared with the flow one step later at the agent's new cell, so it
  includes staleness (the agent moves and the fluid evolves) as well as coupling. Uniform
  conditions have none, which is why A and B are exactly 0.
- The policy also reads other features; only the flow estimate was examined.
- The drag coefficient has no physical calibration, and only the two values 0.5 and 2.0 were
  run.

## Files

All numbers in this README were recomputed from `per_seed.json` by an independent script
(`scripts/audit_lbm_results.py` on `exp/lbm-live-sweep-rescored`): the condition table, the
Wilcoxon and Spearman figures, the mediation point estimates and the cross-evaluation table
reproduce exactly. The mediation bootstrap intervals were not recomputed.


`analysis.md` (full script output), `analysis.json`, `per_seed.json` (every seed, all six
same-fluid conditions and the cross-evaluation cells).
