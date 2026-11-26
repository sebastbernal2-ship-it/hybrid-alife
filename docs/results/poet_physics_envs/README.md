# POET physics-gene environments: pilot

Pilot only. Three seeds on an 8x8 grid. **No claim is made that the physics space is better or
worse than the legacy space**; this run cannot tell them apart (see "What the pilot shows").

## What was added

`POETConfig(physics_genes=True)` appends 14 genes to `EnvironmentGenome.physics`; initialization
and `mutate_environment` handle them. It is off by default. The physics draws use separate
`fold_in` keys, so the legacy parameters and terrain are unchanged for the same key. With the
option off, transfer, archive-transfer, held-out and environment-archive artifacts of a
3-generation smoke run (`poet` and `avida_persistent` tracks, seed 1) hashed the same as
`origin/main` (ignoring the `config` and `source_commit` keys; `config` gains the new fields).
That was a one-off local check, not a CI test.

| gene | bounds (also hard limits) | effect |
|---|---|---|
| viscosity | 0.05 to 0.3 | steady LBM flow under a fixed body force |
| obstacle_radius, _x, _y | 0-0.25 of H, 0.2-0.8 of W, 0.35-0.65 of H | circular solid in the channel |
| curvature_amplitude, _wavenumber | 0-1, 1-4 | serpentine curvature (Dean term and lift) |
| dean_strength | 0 to 0.5 | stylised Dean-like secondary flow added to the LBM field |
| flow_noise_std | 0 to 0.05 | frozen spatial noise, drawn from the world seed |
| resource_abundance | 0.5 to 1.5 | multiplier on initial resources |
| hazard_intensity, hazard_decay | 0.5-2, 0.9-1 | hazard damage multiplier and per-step decay |
| thermo_efficiency | 0.2 to 1 | usable fraction of gathered resource; the rest becomes metabolite |
| signal_diffusion, signal_decay | 0-0.2, 0.8-1 | concentration field (diffusion stencil limit is 0.25) |

Configs may narrow these bounds, not widen them. Invalid genomes (shape, non-finite, out of
bounds, a blocked channel) raise `ValueError`. A mutation that would block the channel keeps the
parent's three obstacle genes. Physics values are traced arguments, so new genomes do not
recompile the evaluator or the world step.

**Low Mach.** The body force gives a nominal Poiseuille peak of `0.005 / viscosity` lattice units
(at most 0.1). `lbm.solve` still rejects anything above 0.3. `physics_lbm_steps` must be at least
`(H-2)^2 / viscosity_low`; this is a time-scale guard, not a measured convergence tolerance.

**Cache.** Native solves run on the host, outside jit, and are kept in an in-process LRU cache
(256 entries) keyed on grid, obstacle mask, viscosity and step count. It is not persisted.

## `lbm_live` under POET jit: not usable

Jitting POET's world step with `flow_solver="lbm_live"` raises no error, but the result is not
live. POET carries only `WorldState`, so `SimState.lbm` is `None` on every call: the fluid is
built and advanced in numpy while jit traces, and the flow is baked in as a constant. It is the
same after 1 or 3 scan steps and ignores the input flow
(`test_lbm_live_solver_is_trace_time_constant_in_poet_scan`). A real live fluid would need a JAX
LBM or a callback with the fluid state in the scan carry. Neither was built. The physics genes use
the steady solver.

## Pilot

Both arms: track `poet`, 3 seeds, 12 generations, population 4, 16-step rollouts,
`configs/poet_physics_pilot.yaml`. Equal budget means equal generations, rollout lengths and
counts of PPO updates (48 per cell) and candidate worlds (12). It does not mean equal wall time or
equal native-solve cost. Initial and final policies are scored on 16 legacy and 16 physics
held-out worlds per seed. The two suites share legacy genes and terrain.

```bash
make -C native
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_poet_physics_pilot.py
```

Raw numbers are in `pilot.json`. Mean held-out reward over all policy/world pairs:

| seed | training arm | legacy suite: initial -> final | physics suite: initial -> final |
|---|---|---|---|
| 0 | legacy | 1.15 -> 2.18 | 0.70 -> 1.72 |
| 0 | physics | 1.15 -> 1.32 | 0.70 -> 1.22 |
| 1 | legacy | 2.95 -> 6.03 | 2.83 -> 6.01 |
| 1 | physics | 2.95 -> 6.03 | 2.83 -> 6.01 |
| 2 | legacy | 2.48 -> 3.35 | 2.14 -> 2.68 |
| 2 | physics | 2.48 -> 3.35 | 2.14 -> 2.68 |

## What the pilot shows

- The pipeline runs end to end in both spaces.
- **The comparison is uninformative.** A follow-up check of final checkpoints (greedy rollouts,
  legacy dynamics, legacy held-out worlds; not part of the script) found each cell's policies
  taking essentially one constant action (1023 or 1024 of 1024 steps). In seeds 1 and 2 both
  arms collapsed to the same action, hence the identical scores. In seed 0 they collapsed to
  different actions.
- "Best policy per world" (`*_best_per_env_mean` in the JSON) is higher for the initial, more
  varied policies than for the final ones in every cell. It selects on the test worlds, so it is
  optimistic.
- All 12 candidates were admitted in every cell. The default minimal criterion (-10) filtered
  nothing, so admission applied no selection to either space.
- Final-over-initial gains in the mean are not evidence of generalization, for the same reason.

## Limits

- On 8x8 the channel has six fluid rows, so obstacles cover few cells and many nearby geometry
  genes give the same mask. LBM accuracy at this resolution was not validated.
- Without an obstacle, viscosity mainly rescales the flow amplitude. Flow moves the forager by
  only `0.02 * flow` per step.
- POET observes only the grid mean of the signal field, so `signal_diffusion` has little or no
  route to a no-Avida policy. `thermo_efficiency` matters only when a policy gathers. Neither
  changed the score of the one untrained policy probed during development.
- `environment_causal_validity` still reports only the legacy genes. Physics genes are covered
  by targeted unit tests instead.
- `thermo_efficiency` is a POET gather-energy rule modelled on the embodied option of the same
  name, not a mass or energy conservation closure.

Next step before any comparison: fix the policy collapse and calibrate the minimal criterion,
then rerun with more seeds.
