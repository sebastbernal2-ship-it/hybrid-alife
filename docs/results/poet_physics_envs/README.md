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

`step_world` raises `ValueError` when `flow_solver="lbm_live"` is traced (jit, scan, vmap).
Without that guard the numpy fluid would be built while jax traces and its flow baked in as a
constant (POET carries only `WorldState`, so `SimState.lbm` is `None` on every call). Eager
calls are unaffected (`test_lbm_live_traced_step_raises_and_eager_stays_live`). A real live
fluid under jit would need a JAX LBM or a callback with the fluid state in the scan carry.
Neither was built. The physics genes use the steady solver.

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

The numbers in this section were measured before the GAE fix below and are left as recorded.

## Why final policies take one action

Measured on this pilot's legacy-space configuration unless stated. The diagnostic, ablation and search
scripts behind items 1 to 6 were one-off and are not committed, so those figures are not reproducible
from the repo.

1. **Greedy evaluation of a near-uniform policy.** After 12 generations, mean action entropy on
   visited states is 1.84 to 2.05 nats (uniform over 8 actions is 2.08). Across states, each logit
   has a standard deviation of about 0.04 to 0.06, while the gap between the two largest mean
   logits is 0.2 to 0.6. The bias therefore decides the argmax in every state. Several untrained
   policies are already constant under argmax for the same reason.
2. **Very small and shared budget.** A cell makes 48 PPO updates of 16 transitions (768 in total),
   split among four lineages. Every generation the worst policy is overwritten by a mutated copy of
   the best, so the final four usually act alike. Arms start from the same policies and share the
   held-out worlds, so two arms that end on the same action score identically.
3. **Ruled out:** evaluator mismatch (the compiled and reference evaluators agree within 5e-7 on
   24 policy/world pairs) and action decoding (8 outputs map to 5 moves plus gather, interact and
   reproduce as documented; actions 5 to 7 reuse moves 0 to 2).
4. **More PPO training still gives a constant.** On one fixed world, 600 updates (three seeds, with
   or without a persistent Adam state) left one greedy action each time, not always the best one.
   On a pool of random worlds, one world per update ended exactly on the best constant action
   (held-out score 3.53). With 64 worlds per update, some runs rose above it (up to 4.10 after
   1000 updates) and others fell below it. Persistent Adam, GAE masking, a larger entropy bonus
   and input standardization did not give a gain that held across seeds. The actor part of this
   ablation re-implemented the PPO loss in a script.
5. **The network and observation are not the limit.** A cross-entropy-method search over the same
   network's weights reached 4.53 to 4.64 on 256 held-out worlds (two runs), against 3.53 for the
   best constant and 6.48 for a per-world oracle that picks the best constant.
6. **The reward favours constants.** Gather, interact and move are paid or charged by action
   choice, so a constant is a strong local optimum. The observation has goal distance but no goal
   coordinates or direction. A goal within 0.12 of the start pays +1 again after every reset; for such
   worlds in the 256-world probe, constant action 6 averaged 11.8 against 2.8 elsewhere. How much
   each of these causes PPO to stop at a constant was not isolated.

**Status: the collapse is not fixed.** No PPO change tried here reliably produces state-dependent
policies. Fixing it likely needs a design change (goal direction in the observation, a reward that
cannot be farmed, a much larger batched budget, or a different optimizer). None is made here.
POET comparisons from this module should not be read as comparisons of learned behaviour.

## One bug fixed: GAE crossed episode resets

Success and failure reset the forager, but `_gae` bootstrapped from the next episode. This is now
cut at each reset, **by default**, because it is a bug rather than an option
(`test_advantages_do_not_bootstrap_across_episode_resets`). Training runs from current code will
not reproduce earlier recorded numbers; check out the recorded source commit for that. Frozen-policy
evaluation is unaffected. It does not fix the collapse.

## Calibrated minimal criterion and re-pilot

The default stays -10. `configs/poet_physics_repilot.yaml` sets 2.8, the 25th percentile of 120
candidate scores from admit-all runs of both arms on seeds 100 to 104. The re-pilot uses seeds 0 to 2.

```bash
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_poet_physics_pilot.py \
  --calibrate --seeds 100 101 102 103 104
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_poet_physics_pilot.py \
  --config configs/poet_physics_repilot.yaml --out docs/results/poet_physics_envs/repilot.json
```

Raw numbers, with per-policy modal-action share and entropy, are in `repilot.json`.

| seed | arm | admitted of 12 | legacy suite: initial -> final | physics suite: initial -> final |
|---|---|---:|---|---|
| 0 | legacy | 5 | 1.15 -> 2.89 | 0.70 -> 2.95 |
| 0 | physics | 0 | 1.15 -> 2.18 | 0.70 -> 1.72 |
| 1 | legacy | 11 | 2.95 -> 5.66 | 2.83 -> 5.72 |
| 1 | physics | 11 | 2.95 -> 6.12 | 2.83 -> 6.03 |
| 2 | legacy | 4 | 2.48 -> 3.62 | 2.14 -> 2.75 |
| 2 | physics | 7 | 2.48 -> 2.91 | 2.14 -> 2.39 |

- The criterion now discriminates: 38 of 72 candidates were admitted, from 0 to 11 per cell. Because it is a
  fixed score threshold, rejection varies a lot by seed. It filters worlds by how the current best
  policy scores, which for a near-constant policy mostly reflects the world's reward level. It does
  not add an upper bound, so it does not remove farmable near-start goals.
- Final greedy modal-action share (seed 0: 1.00; seed 1: 0.58 to 1.00; seed 2: 0.84 to 1.00). Final entropy is 1.96 to 2.05 nats.
  Policies are still near-uniform, and mostly constant under argmax. Seed 1 varies more, which is
  not evidence of a fix.
- Three seeds. This run supports no comparison between the two spaces.
