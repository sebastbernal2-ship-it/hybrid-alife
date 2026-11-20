## Reproduction check against the recorded min_delta (max |difference| over 15 seeds)
- A: 0.00e+00
- B: 0.00e+00
- C: 0.00e+00
- D: 0.00e+00
- E: 0.00e+00
- F: 0.00e+00

## Estimate error by condition (evolved genomes, 4 held-out diagonals, 15 seeds)
| Condition | angle mean (deg) | angle median over seeds | > 45 deg | flow-vs-law angle (deg) | flow magnitude dev | median min_delta | vs B: angle diff (Wilcoxon p) |
|---|---|---|---|---|---|---|---|
| A legacy uniform | 0.00 | 0.00 | 0.000 | 0.00 | 0.000 | 0.393 | +0.00 (0/15 higher, p=1) |
| B live, no coupling | 0.00 | 0.00 | 0.000 | 0.00 | 0.000 | 0.393 | +0.00 (0/15 higher, p=1) |
| C monopole 0.5 | 1.08 | 1.10 | 0.000 | 3.87 | 0.176 | 0.328 | +1.08 (15/15 higher, p=6.1e-05) |
| D dipole 0.5 | 1.22 | 1.17 | 0.000 | 2.16 | 0.065 | 0.325 | +1.22 (15/15 higher, p=6.1e-05) |
| E monopole 2.0 | 4.14 | 3.86 | 0.003 | 13.33 | 0.686 | 0.255 | +4.14 (15/15 higher, p=6.1e-05) |
| F dipole 2.0 | 6.33 | 6.53 | 0.007 | 9.66 | 0.271 | 0.418 | +6.33 (15/15 higher, p=6.1e-05) |

## Does estimate error predict min_delta?
| Pool | Spearman rho (p) | rho after removing per-condition means (p) |
|---|---|---|
| all six conditions (n=90) | -0.15 (0.169) | -0.22 (0.0373) |
| coupled C-F (n=60) | -0.12 (0.366) | -0.30 (0.0213) |

| Condition (n=15) | Spearman rho (p) |
|---|---|
| C monopole 0.5 | -0.07 (0.81) |
| D dipole 0.5 | -0.32 (0.243) |
| E monopole 2.0 | -0.05 (0.85) |
| F dipole 2.0 | -0.59 (0.0218) |

## Mediation check: how much of the min_delta gap against B runs through a mediator
Seed-resampled bootstrap (4000), 95% interval; pooled B + condition, OLS. total = gap,
direct = gap after adding the mediator, indirect = total - direct.

| Condition | Mediator | total gap | direct | indirect [95% CI] | mediator coef |
|---|---|---|---|---|---|
| E | angle_mean | -0.134 | -0.102 | -0.032 [-0.429, +0.310] | -0.0078 |
| E | law_angle_mean | -0.134 | -0.219 | +0.085 [-0.620, +0.471] | +0.0064 |
| E | magnitude_dev_mean | -0.134 | -0.215 | +0.081 [-0.343, +0.388] | +0.1181 |
| F | angle_mean | +0.001 | +0.237 | -0.236 [-0.428, -0.097] | -0.0373 |
| F | law_angle_mean | +0.001 | +0.320 | -0.319 [-0.603, -0.097] | -0.0331 |
| F | magnitude_dev_mean | +0.001 | +0.402 | -0.401 [-0.747, -0.130] | -1.4778 |

## Cross-evaluation: genomes trained in one fluid, evaluated in another
| Train -> eval | median min_delta | mean angle (deg) | vs same-fluid B->B: paired mean min_delta diff (Wilcoxon p, worse/15) |
|---|---|---|---|
| B -> B | 0.393 | 0.00 | +0.000 (p=1, 0/15) |
| E -> E | 0.255 | 4.14 | -0.134 (p=0.0125, 13/15) |
| F -> F | 0.418 | 6.33 | +0.001 (p=0.978, 7/15) |
| B -> E | 0.359 | 4.02 | -0.093 (p=0.00116, 13/15) |
| B -> F | 0.371 | 6.18 | -0.016 (p=0.33, 10/15) |
| E -> B | 0.372 | 0.00 | -0.018 (p=0.639, 9/15) |
| E -> F | 0.375 | 5.83 | -0.033 (p=0.454, 10/15) |
| F -> B | 0.427 | 0.00 | +0.009 (p=0.72, 6/15) |
| F -> E | 0.322 | 3.99 | -0.084 (p=0.0181, 11/15) |
