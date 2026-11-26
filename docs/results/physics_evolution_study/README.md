# Physics evolution study

Do the opt-in physics features change evolutionary outcomes? Design, arms, regime and the
list of what had been seen beforehand: [PREREGISTRATION.md](PREREGISTRATION.md).

Not fully blind: before pre-registration, the original ten conditions were smoke-tested at seed
1000 (3 generations, old base config) and their feature outputs were seen. The harsh regime was
then chosen by calibrating the baseline arm only (seeds 1000-1007).

## Primary results (seeds 0-14, paired against the baseline)

| Arm | Outcome | Arm / baseline mean | Mean difference [95% CI] | dz | p | Holm p | Reading |
|---|---|---|---|---|---|---|---|
| thermo | embodied births | 15.9 / 104.7 | -88.8 [-114.6, -67.9] | -1.85 | 0.00006 | 0.0004 | difference, caused by collapse (below) |
| signaling | final-quarter emit rate | 0.432 / 0.504 | -0.071 [-0.208, 0.060] | -0.26 | 0.35 | 0.69 | null |
| stokes | final body radius | 1.119 / 1.196 | -0.077 [-0.180, 0.005] | -0.40 | 0.14 | 0.58 | null |
| avida_energy (0.002 vs 0.001) | final Avida genome length | 26.4 / 20.1 | 6.25 [0.92, 11.54] | 0.57 | 0.044 | 0.22 | null after Holm |
| lbm_dipole | embodied births | 83.7 / 104.7 | -21.0 [-52.9, 6.7] | -0.35 | 0.21 | 0.63 | null |
| combined | final embodied lineage depth | no survivors | not estimable (0 pairs) | - | - | - | untestable |

The untestable contrast stays in the Holm family of six (entered as p = 1).

"Null" means no difference was detected. With 15 pairs and a worst-case Holm alpha of 0.05/6,
80% power needs |dz| of about 1.03 (0.78 unadjusted), so small and moderate effects, including
ones the size of the observed point estimates, are not ruled out.

## Thermo and combined: no living embodied population

In both arms no embodied agent was alive at the end of any of the 300 seed-generations, with no
embodied births in the last 10 generations. Selection revived slots and `reseed_embodied` fired
every generation. The thermo result therefore says that `thermo_efficiency` 0.5 is not viable
in this regime (calibrated without it). It says nothing about how thermo changes adaptation.
The combined arm includes thermo, so its embodied outcomes are uninformative too; that thermo
alone explains its collapse is an inference. The other features were not tested in
combination without thermo.

## Avida energy cost

The on-versus-off contrast was dropped before the confirmatory runs: without a cost, no slot
ever frees up. `avida_nocost` showed zero Avida births and deaths in all 15 seeds. The tested
contrast is a dose (0.002 versus 0.001). Its primary outcome is null after Holm. Exploratory,
unadjusted: the higher cost gave more Avida births (459 versus 91; difference 368, interval
[169, 605]) and a lower minimum alive count (19.3 versus 29.7). Those mostly reflect the cost's
direct effect on deaths.

## Exploratory controls

Unadjusted, from many comparisons; see `analysis.json`. No claim rests on them.

- `ledger` versus baseline: lower `av_merit`, larger `body_radius`, lower
  `emb_min_alive_end` (unadjusted p < 0.05). It also changes reproduction, so none of this is
  attributable to bookkeeping or to the slot-0 fix specifically.
- `lbm_oneway` versus baseline: lower embodied lineage depth (unadjusted p < 0.05). No
  outcome separated `lbm_dipole` from `lbm_oneway` at that level.
- `legacy_divide` versus baseline: no outcome at that level.

## Regime caveats

The baseline had living populations (no generation ended with fewer than 3 alive) and averaged
104.7 embodied births, 33.2 of them in the last 10 generations. All its Avida turnover (91.1
births per seed) fell in the last 10 generations, after the initial energy buffer drained, so
Avida outcomes reflect a short window of turnover. `stokes` had 2 and `ledger` 1
seed-generations with fewer than 3 embodied agents alive. Births and deaths exclude
end-of-generation revivals and reseeding.

## Checks

- Plan reviewed by Codex (read-only) before the pre-registration commit. It led to wording
  fixes on the test's symmetry assumption and on what the birth counts include.
- Codex independently recomputed the six primary contrasts from the per-seed JSONs without
  reading the analysis script or `analysis.json`. Means, differences, dz, p and Holm p matched.
  Bootstrap interval ends differed slightly (independent resampling). It also confirmed that
  per-seed totals equal the sums of the generation records.
- Runs used the simulator at `bd6d84c`. After merging `main` at `bdd818c`, seeds 0 and 1 of
  `baseline`, `thermo`, `lbm_dipole` and `combined` reproduced byte for byte; other arms and
  seeds were not rerun.

## Files and reproduction

`<arm>/seed_<n>.json` (per run), `analysis.json`, `calibration.json`,
`calibration_basal_sweep_prior.txt`.

```
make -C native
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_physics_evolution_study.py \
  --conditions baseline thermo signaling stokes avida_energy lbm_dipole combined \
  ledger legacy_divide lbm_oneway avida_nocost \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 --out docs/results/physics_evolution_study
PYTHONPATH=src python scripts/analyze_physics_evolution_study.py docs/results/physics_evolution_study
```
