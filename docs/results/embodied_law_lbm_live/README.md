# Flow-head law study with a live, agent-coupled fluid

Status: **bounded pilot, 15 seeds per condition. Strong monopole coupling (drag 2.0) lowers
the margin by 0.13 on 13 of 15 seeds (paired Wilcoxon p = 0.012) and fails 5 seeds; a dipole
at the same strength has no effect. Weaker coupling (drag 0.5) is unresolved. Not a claim.**

## Question

Does two-way coupling to a live lattice-Boltzmann fluid, as a monopole (agents push the
fluid with their thrust) or a force dipole (zero net force, as a Stokes swimmer exerts),
change whether the flow-head organisms learn the imposed law, compared with the uncoupled
controls? Protocol: the recorded flow-head study (8 generations, 16 agents, 8 steps per
generation, 14x14, eval 12 steps, displacement reward 5.0, unit-flow feature, flow head,
min-alignment fitness with 6 frozen steps, rank replacement, 25% immigrants, margin gate
0.2, reversed-law floor 0.2, held-out diagonals, minimum over directions decides).

## Design: how a live fluid carries the law

The study imposes the law by overwriting `world.flow`, but `lbm_live` refreshes
`world.flow` from the solver every step, so the two cannot be combined directly. Design:

- **The law is the mean velocity of a fully periodic, wall-free LBM box** (`lbm_box:
  periodic`, `LiveLBM.uniform`). A uniform flow is an exact steady state of periodic
  Navier-Stokes, so the fluid is initialised at its equilibrium and **needs no relaxation
  steps**. Flow units are `velocity / lbm_peak_speed`; peak speed 0.25 maps the imposed flow
  magnitude 0.2 to a lattice velocity of 0.05 (Mach about 0.09).
- **The evaluator reads the actual live local flow.** `step_world` refreshes `world.flow`
  from the fluid each step; `_local_flows` and the agents' advection use it.
- **Re-imposing an unchanged law is a no-op** (the study re-imposes after every step in
  legacy mode). Only a new direction or the reversed-law control restarts the fluid at the
  new uniform flow, with a fresh fluid per evaluation rollout.
- **Mean-flow relaxation** (`lbm_law_relax_steps = 20` LBM steps, two world steps): a uniform
  acceleration `(target - mean) / 20` plays the role of the wall friction or pump that sets a
  channel's flow. Without it a net monopole force accelerates a periodic fluid without
  bound. Local perturbations (agent wakes) are not suppressed, only the mean is held.
  The controller is identical in every live condition.
- **Settledness:** `tests/test_law_carrier_live.py` checks that the fluid at rest in the law
  stays within 1e-6 of it for 600 LBM steps, that its direction has cos > 0.99 and its speed
  equals the law for 4 cardinals and 4 diagonals and for each reversed law, that the
  uncoupled live flow equals the legacy uniform flow to float32 (2e-7), that the controller
  holds a net force at the expected steady offset `force x relax`, and that a dipole leaves
  the mean flow where a monopole shifts it.

Coupling strength, measured on random-genome populations over 12 steps (`lbm_agent_drag`
against the 0.2 law): at 0.5 the local flow departs from the law by up to 34% (monopole) or
21% (dipole); at 2.0 by up to 123% / 86%. The mean flow stays within 4% of the law
throughout (monopole) or exactly on it (dipole). So 0.5 is a moderate and 2.0 a strong local
perturbation.

## Results (15 seeds each, seeds 0-14)

| Condition | Pass | Median min_delta | Failing seeds (min_delta / min_reversed) |
|---|---|---|---|
| A  legacy uniform law (baseline) | 14/15 | 0.393 | 1 (0.193 / 0.411) |
| B  live fluid, coupling off (drag 0) | 14/15 | 0.393 | 1 (0.193 / 0.411) |
| C  live, monopole, drag 0.5 | 15/15 | 0.328 | none |
| D  live, dipole, drag 0.5 | 14/15 | 0.325 | 9 (0.134 / 0.378) |
| E  live, monopole, drag 2.0 | 10/15 | 0.255 | 4 (-0.047 / 0.520), 6 (0.032 / 0.266), 7 (0.171 / 0.437), 9 (0.142 / 0.586), 14 (0.125 / 0.303) |
| F  live, dipole, drag 2.0 | 14/15 | 0.418 | 8 (0.168 / 0.342) |

**Carrier control (A against B).** B reproduces A **exactly**: every one of the 15 seeds has
identical min_delta (maximum difference 0.0). The live-fluid carrier is not a confound, so
differences below can be attributed to the coupling.

Pass-count comparisons (Fisher exact, uncorrected): E against A or B p = 0.169; C against E
p = 0.042; every other pair p >= 0.17 (most 1.0). Pass counts alone resolve nothing against
the baseline. Because B is identical to A, the seeds are paired, and a paired Wilcoxon on
min_delta (more power than a pass/fail count) is the better test:

| Against baseline A | Mean change in min_delta | Seeds worse / better | Wilcoxon p |
|---|---|---|---|
| C monopole 0.5 | -0.034 | 10 / 5 | 0.49 |
| D dipole 0.5 | -0.039 | 9 / 6 | 0.36 |
| E monopole 2.0 | **-0.134** | **13 / 2** | **0.012** |
| F dipole 2.0 | +0.001 | 7 / 8 | 0.98 |

Monopole against dipole at the same strength: at 2.0, paired mean +0.135 for the dipole,
p = 0.003; at 0.5, -0.005, p = 0.89. Reversed-law alignment (min_reversed) does not move
in any condition (mean change within 0.03, all p > 0.6).

## Reading

- **Resolved:** strong monopole coupling (E) lowers the margin, consistently across seeds
  (13 of 15 worse, mean -0.13, p = 0.012; 4 comparisons against A, so p = 0.048 after
  Bonferroni), and the dipole at the same strength (F) does not (p = 0.003 for E against F).
  The reversed-law floor is unaffected, so E's failures are margin failures (seed 4 fails
  hard, -0.047), not a switch to a fixed action.
- **Not resolved:** moderate coupling (C, D). Both lower median min_delta by about 0.06 but
  p = 0.36 to 0.49, and C passes 15/15. Do not read the C = 15/15 as improvement.
- **Plausible mechanism, untested:** the monopole's net momentum injection and its
  persistent wakes change the local flow that the proprioceptive estimate
  `(velocity - thrust)` reads, so a head tuned on a constant flow sees a noisier signal; the
  dipole's far field decays faster and conserves momentum. Not measured here (no estimate
  error was logged).
- Multiplicity: 4 coupled conditions against A plus pairwise comparisons were examined.
  Only E against A and E against F are below 0.05, and E against A is borderline after
  correction.

## Caveats

- **Coupling strength is a free parameter**, and the drag coefficient has no calibration to
  a physical swimmer. The monopole is the unphysical model (thrust treated as an external
  push); its drag 2.0 result says that strong monopole forcing degrades learning, not that
  real swimmers would.
- **A small periodic box** (14x14, 16 agents) with a mean-flow controller. The result is
  specific to it. A wall-bounded channel carrier was not tested here.
- The dipole length is quantised to even cell separations (1219009), so dipole strength is
  not smoothly tunable.
- Drag values 0.5 and 2.0 only; no 0.05 or intermediate sweep. 15 seeds per arm.
- Training and evaluation both run in the live fluid; no train-uncoupled, test-coupled
  transfer condition.

## Next

Log the proprioceptive flow-estimate error per condition to test the mechanism; sweep drag
between 0.5 and 2.0 to locate the onset for the monopole; repeat in a wall-bounded channel
carrier; and add train-uncoupled, evaluate-coupled transfer.

Reproduce (from the worktree, `PYTHONPATH=src JAX_PLATFORMS=cpu`):
`python scripts/run_embodied_law_study.py --seeds 0 1 2 3 4 --generations 8 --population 16
--steps-per-generation 8 --size 14 --eval-steps 12 --displacement-reward 5.0
--unit-flow-features --flow-head --fitness-mode min_alignment --fitness-steps 6
--rank-replacement --immigrant-fraction 0.25 --law-carrier live --lbm-drag 2.0
--lbm-coupling monopole`
