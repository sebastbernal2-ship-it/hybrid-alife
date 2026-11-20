# Why does the monopole-2.0 fluid lower min_delta? Self-contamination of the baseline

Status: **bounded diagnostic, 15 seeds. Most of the monopole-2.0 loss is a measurement
effect: each agent's own wake, which the monopole pushes along its own thrust, inflates the
alignment of the *initial* (random) policies far more than that of the evolved ones, so the
evolved-minus-initial margin shrinks. Scoring without the agent's own wake removes the loss
for B-trained genomes. A smaller residual for E-trained genomes is unresolved. Not a claim.**

Parents: `../README.md` (monopole drag 2.0 lowers min_delta by 0.134), `../estimate_error/`
(estimate error does not explain it; the loss is mostly an evaluation-time effect of that
fluid), `../drag_sweep/` on `exp/lbm-live-sweep` (gradual dose-response; dipole flat).

## Question

Which feature of the monopole fluid lowers the measured alignment margin? Candidates:
(H1) the scoring target is perturbed away from the law; (H2) each agent is scored against a
field it just perturbed (self-contamination); (H3) other agents' wakes perturb the target;
(H4) the perturbed flow changes the agents' own motion and policy output.

## Design

The score is `min over diagonals of (evolved - initial)` where each is the mean cosine
between the commanded thrust (`last_action[:, :2]`) and the live local flow at the agent's
cell, in the *same* fluid. So a change in either term moves it.

- **Shadow fluids.** During each coupled rollout, one extra `LiveLBM` per agent is advanced in
  lockstep, forced only by that agent (`agent_force_field(..., alive=one-hot)`, factored out of
  the runner without changing behaviour; the per-agent fields sum exactly to the full field,
  asserted at every step). With the mean-flow controller and small Mach number the fluid is
  near-linear, so an agent's own wake at its own cell is `shadow_i - law`, and the flow
  without self-contamination is `flow - (shadow_i - law)`. Independent check: the sum of the
  other agents' shadows (`noself_sum`).
- **Post-hoc rescoring of identical rollouts** against four targets: the official local flow;
  the imposed law (a function of thrust only); the flow without the agent's own wake
  (`noself`, = law + others' wakes); law plus only the own wake (`selfonly`). Rescoring is
  arithmetic on one set of rollouts, so the policy's behaviour is held fixed across targets.
  This separates target effects (H1-H3) from behaviour effects (H4).
- **Genomes.** B-trained and E-trained evolved genomes plus the shared initial genomes
  (identical across conditions per seed; asserted), retrained from scratch for seeds 0-14
  because earlier caches were lost. Fluids: B (no coupling), E (monopole drag 2.0), F (dipole
  drag 2.0). Same protocol as the parent studies.
- **Cross-replay control.** The E-fluid flow series recorded from one population is replayed
  with the agents *not* forcing, and the other population is scored in it. (Replaying a
  population's own series is a tautology: its behaviour depends only on the flow it sees, so
  the trajectory repeats exactly. Verified: the first replay design returned numbers identical
  to the full rollout to every digit, so it was dropped.) Own wake against foreign wake then
  isolates the contamination from the dynamics.
- **Statistics.** Paired by seed; Wilcoxon; seed-bootstrap 95% CIs; Holm over the 8 primary
  fluid-E tests (2 genome sets x official/law/noself/selfonly).
- **Not feasible without invasive changes:** a *dynamic* self-only or others-only fluid (each
  agent reading its own fluid) would need per-agent solvers inside the sim step. The
  shadow decomposition replaces it for the target; it cannot separate H3 from H4 in dynamics.

Reproduction check: median min_delta equals the parent values (B in B 0.393, B in E 0.359,
E in E 0.255), the rescored official score equals `evaluate_frozen_alignment` to 1e-9, and
adding shadows does not change a rollout (`tests/test_mechanism_rescore.py`).

## Results

### Mean alignment (15 seeds x 4 diagonals)

| Fluid | Genomes | official | law | noself | selfonly | noself_sum |
|---|---|---|---|---|---|---|
| B | B-trained | 0.612 | 0.612 | 0.612 | 0.612 | 0.612 |
| B | initial | 0.056 | 0.056 | 0.056 | 0.056 | 0.056 |
| E | B-trained | 0.671 | 0.569 | 0.591 | 0.678 | 0.592 |
| E | initial | 0.239 | 0.073 | 0.049 | 0.267 | 0.050 |
| F | B-trained | 0.609 | 0.600 | 0.607 | 0.602 | 0.607 |
| F | initial | 0.066 | 0.052 | 0.057 | 0.061 | 0.057 |

The own wake raises alignment (sign checked, not assumed): in E the wake at an agent's cell
points along its own thrust (cosine 0.93 evolved, 0.89 initial), so any policy with a
steady thrust scores higher against the flow it has just stirred. The initial policies gain
+0.183 (0.056 to 0.239); the evolved ones gain +0.059 (0.612 to 0.671), because they have
less headroom. With the own wake taken out of the target (`noself`) the initial policies
return to 0.049, the baseline. In the dipole fluid the own wake is 5 times smaller and
uncorrelated with thrust (cosine -0.21 evolved, +0.11 initial), and nothing moves.

### min_delta by scoring target, fluid E against the same genomes in fluid B

| Genomes | Scored | Median | Mean change [95% CI] | Seeds worse/better | p | Holm p (8) |
|---|---|---|---|---|---|---|
| B-trained | official | 0.359 | -0.093 [-0.128, -0.056] | 13/2 | 0.0012 | 0.0035 |
| B-trained | law | 0.333 | -0.063 [-0.090, -0.040] | 14/1 | 0.0001 | 0.0009 |
| B-trained | **noself** | 0.393 | **-0.015 [-0.049, +0.017]** | 8/7 | 0.60 | 0.60 |
| B-trained | selfonly | 0.311 | -0.128 [-0.166, -0.091] | 14/1 | 0.0001 | 0.0009 |
| E-trained | official | 0.255 | -0.116 [-0.151, -0.084] | 14/1 | 0.0001 | 0.0009 |
| E-trained | law | 0.275 | -0.098 [-0.131, -0.065] | 14/1 | 0.0001 | 0.0009 |
| E-trained | **noself** | 0.340 | **-0.036 [-0.069, -0.003]** | 10/5 | 0.073 | 0.146 |
| E-trained | selfonly | 0.187 | -0.165 [-0.201, -0.126] | 15/0 | 0.0001 | 0.0005 |

`noself_sum` (independent estimate of the same target) agrees: -0.015 (p = 0.64) and -0.035
(p = 0.083). In the dipole fluid F no scoring target gives a resolved change for either genome
set (all p >= 0.27).

Share of the loss removed by scoring without the own wake: 0.84 [0.57, 1.28] for B-trained,
0.69 [0.47, 0.97] for E-trained (seed bootstrap).

### Which side of the margin moves (B-trained genomes, fluid E against fluid B)

| Scoring | Evolved alignment change [CI] | Initial alignment change [CI] |
|---|---|---|
| official | +0.059 [+0.040, +0.083] | **+0.183 [+0.171, +0.196]** |
| law | -0.043 [-0.057, -0.029] | +0.017 [-0.003, +0.037] |
| noself | -0.022 [-0.039, +0.001] | -0.008 [-0.020, +0.006] |

### Cross-replay (E-fluid wake replayed, agents not forcing; B-trained genomes)

| Genome set | In its own wake | In a foreign wake | Difference [95% CI] | p |
|---|---|---|---|---|
| initial | 0.239 | 0.058 | +0.181 [+0.160, +0.203] | 0.0001 |
| B-trained | 0.671 | 0.616 | +0.056 [+0.038, +0.074] | 0.0002 |

The initial policies' inflation disappears when the wake in the field is not their own, even
though a wake of the same size is still present, so the effect is contamination, not the
presence of a perturbation. The margin follows the same pattern: in the wake recorded from
the *evolved* population (evolved genomes in their own wake, initial genomes in a foreign one)
the official min_delta rises by +0.060 against fluid B (p = 0.002); in the wake recorded from
the *initial* population (initial in their own wake, evolved in a foreign one) it falls by
-0.157 (15 of 15 seeds worse, p = 0.0001).

### Wake size (fluid E)

| Genomes | Own wake at own cell | Others' wake at cell | Total perturbation | Superposition residual, mean / max |
|---|---|---|---|---|
| B-trained | 0.078 | 0.087 | 0.142 | 0.068 / 0.139 |
| initial | 0.077 | 0.060 | 0.095 | 0.052 / 0.097 |

Flow units, law magnitude 0.2. The residual is `|sum of shadow wakes - full wake| / |full wake|`
over the field, a measure of how far the decomposition departs from linearity.

## Reading

Supported:
- **H2, self-contamination, is the main mechanism.** Scoring without the own wake removes the
  B-trained loss (-0.093 to -0.015, p = 0.60); the own wake alone reproduces it and more
  (-0.128); the cross-replay shows the baseline inflation depends on the wake being the
  policy's own. It acts through the baseline: the initial policies' alignment rises three
  times as much as the evolved ones'.
- The monopole is therefore *not* making evolved organisms worse at following the flow they
  actually experience: their alignment with the local flow rises (0.612 to 0.671). The margin
  falls because the control gets credit for stirring its own field.
- The dipole, with a small uncorrelated own wake, leaves the margin intact, matching the
  parent result.

Ruled out as the main driver:
- **H3, others' wakes**, and **H1, a perturbed target in itself**: with only the others' wake
  left in the target (`noself`) the B-trained margin is unchanged.

Small and unresolved:
- **H4, behaviour change.** The evolved policies' thrust aligns less with the *law* in fluid E
  (-0.043, p = 0.0002): they track the real local flow, wakes included. That costs alignment
  with the law, not with the flow they are scored on, and `law` scoring is not the study's
  metric. On the `noself` target the evolved side moves by -0.022 (p = 0.008 uncorrected).
- **The E-trained remainder.** After removing the own wake, E-trained genomes still show
  -0.036 [-0.069, -0.003] (p = 0.073, Holm 0.146). It may be others' wakes, behaviour, or
  chance; this study cannot say.

What this means for the earlier results: "monopole drag 2.0 lowers the learning margin"
should be read as "monopole drag 2.0 inflates the baseline against which learning is
measured". Using the `law` target instead still shows a loss, because a flow-follower is
penalised for tracking wakes, so neither target is a clean measure of learning under
coupling. A margin that is robust to this would score both populations against a target the
agents cannot influence, or against a no-coupling copy of the same trajectory.

## Caveats

- 15 seeds, one drag value (2.0), one carrier (14x14 periodic box with mean-flow controller),
  uncalibrated drag coefficient, monopole is the unphysical model.
- The `noself` target is a model-based estimate (shadow superposition). The superposition
  residual is 7% on average and up to 14% of the full wake; the independent `noself_sum`
  estimate and the cross-replay, which does not rely on it, point the same way.
- Rescoring holds behaviour fixed, so it cannot separate others' wakes from behaviour in
  dynamics (H3 against H4); a dynamic version needs per-agent fluids inside the step.
- The cross-replay foreign wake comes from a population with different positions, so it is not
  an exact counterfactual for the same trajectory.
- The drag sweep's gradual trend is consistent with a self-wake baseline that grows with drag,
  but that was not tested here.
- The Holm family is the 8 fluid-E tests listed; the dipole, wake and side-of-margin tables
  are descriptive.

## Files

Every figure in the tables above (mean alignments, the min_delta changes with Wilcoxon p,
Holm adjustments, the cross-replay own-versus-foreign-wake differences, the wake-size and
superposition-residual statistics, the cosines with thrust) was recomputed from
`eval_per_seed/` and `replay_per_seed/` by an independent script
(`scripts/audit_lbm_results.py` on `exp/lbm-live-sweep-rescored`) and reproduces. Bootstrap
intervals agree to within resampling noise (for example the share removed for B-trained
genomes comes out 0.84 [0.57, 1.25] against the 1.28 printed above). The rescoring of the
drag sweep against targets the agents cannot influence is `../drag_sweep_rescored/` on
`exp/lbm-live-sweep-rescored`.


`analysis.txt` (full script output), `analysis.json`, `eval_per_seed/` and `replay_per_seed/`
(every seed). Scripts: `scripts/measure_monopole_mechanism.py` (train / eval / replay
phases), `scripts/analyze_monopole_mechanism.py`. Tests: `tests/test_mechanism_rescore.py`.
Genome caches (not committed) were written to `outputs/mechanism/` in the main clone.
