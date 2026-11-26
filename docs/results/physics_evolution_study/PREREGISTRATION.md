# Pre-registration: do opt-in physics features change evolutionary outcomes?

Committed before any confirmatory run (seeds 0-14).

## What was already seen (disclosure)

- All the original ten conditions were smoke-tested at seed 1000 for 3 generations, with the
  old base config (basal cost 0.5, no base Avida cost). Their feature outputs were seen.
  Primary outcomes below were therefore not chosen blind to every feature output.
- Calibration used only the baseline arm with overrides, seeds 1000-1007, disjoint from the
  confirmatory seeds. No feature arm was run during calibration after the smoke test.

## Regime (calibration: `calibration.json`, `calibration_basal_sweep_prior.txt`)

- Embodied `basal_metabolic_cost` 1.1. At 0.5 and 0.8 late turnover was 0-3 events per seed.
  At 1.1 the baseline had 81-201 births over 20 generations and 5-110 in the last 10
  (seeds 1000-1007). Within-generation minima were 2-14 of 32 alive at generation end, so
  `reseed_embodied` can fire in some seeds. This regime is harsh, not stationary.
- Avida: with no instruction cost every slot stays occupied, so there are no deaths and no
  births (division needs a free slot). A feature-off versus feature-on contrast of the Avida
  energy economy therefore has a frozen reference and is **dropped**. Every arm, baseline
  included, uses `instruction_energy_cost` 0.001 (`energy_efficiency` 0.5), which gave 45-135
  births per seed with 31 or more of 32 alive at every generation end. All of that turnover is
  in the last 10 generations, after the initial energy buffer drains.
- The `avida_energy` arm is a **dose contrast**: cost 0.002 versus 0.001. In calibration
  (basal 1.0) 0.002 reduced some seeds to 4-11 alive without extinction.
- `divide_scatter_fix` is on in every arm except `legacy_divide`.
- 20 generations of 50 steps, 16x16 world, 32 + 32 slots. End-of-generation selection revives
  dead embodied slots, as in `run_experiment`.

## Arms and the confirmatory family (6 tests)

Each arm is compared with the baseline, paired by seed (seeds 0-14).

| Arm | Change from baseline | Primary outcome |
|---|---|---|
| thermo | `thermo_efficiency` 0.5 | `emb_births` (total) |
| signaling | `chemical_signaling` + `signal_content` | `emit_rate_final_quarter` |
| stokes | `stokes_mechanics` + `aspect_drag` | `body_radius` (final, alive mean) |
| avida_energy | Avida cost 0.002 | `av_genome_length` (final, alive mean) |
| lbm_dipole | `lbm_live`, drag 0.5, dipole coupling | `emb_births` (total) |
| combined | all five changes above | `emb_lineage_depth` (final, alive mean) |

## Exploratory controls (no confirmatory claims)

- `ledger`: `energy_ledger` alone. It also changes embodied reproduction (slot-0 scatter fix),
  so it is a confound, not pure bookkeeping. The baseline keeps the legacy behaviour.
- `legacy_divide`: `divide_scatter_fix` off.
- `lbm_oneway`: `lbm_live` with agent drag 0; also compared with `lbm_dipole`. `lbm_dipole`
  versus baseline mixes a change of flow solver with two-way coupling.
- `avida_nocost`: no Avida energy economy (descriptive; expected zero Avida turnover).
- All non-primary outcomes in the six confirmatory arms.

## Analysis (`scripts/analyze_physics_evolution_study.py`, committed with this file)

- Test: two-sided paired sign-flip test on the mean paired difference, enumerating all 2^15
  sign patterns. It assumes paired differences are symmetric about zero under the null. Arms
  are not randomised within a seed, so "exact" refers to the enumeration under that assumption.
- Births and deaths are within-step events only (a slot that is alive with a new lineage id,
  and the matching alive-count balance). End-of-generation selection revivals and
  `reseed_embodied` / `reseed_avida` replacements are not counted.
- Pairing shares the seed and initial draws; trajectories diverge once an arm's physics
  differs, so pairing may reduce little variance.
- Holm correction over the six primary tests, family-wise alpha 0.05.
- 95% percentile bootstrap interval (20000 resamples of pairs, fixed RNG seed). Intervals are
  per-contrast and not multiplicity adjusted.
- A pair with a missing outcome (no survivors) is dropped and its count reported.
- A primary test with Holm p >= 0.05 is reported as a null: "no difference detected", not
  "no effect".
- Power: n = 15. At the worst-case Holm level 0.05/6 the paired t approximation gives 80%
  power only for standardised differences |dz| of roughly 1 or more (exact value in
  `analysis.json`). Smaller effects will mostly be missed.
- No seeds are added, removed or rerun based on results. Crashed runs are rerun with the same
  seed and reported.

## Limits

Short runs in a small world, one regime chosen on the baseline. Results do not transfer to
other costs, sizes or horizons. A difference in a primary outcome is not by itself evidence of
adaptation: a feature can change a count or a mean through its direct physical effect.
