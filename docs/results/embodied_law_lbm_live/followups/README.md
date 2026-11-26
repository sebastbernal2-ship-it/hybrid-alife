# Follow-ups: lattice-speed guard and the own-wake-removed target

Status: **checks on the parent study, not new claims.** Both are post hoc.
Script: `scripts/measure_lbm_live_followups.py`; numbers: `analysis.txt`
(`python scripts/analyze_lbm_followups.py`).

## Lattice speed at drag 2.0 against the 0.3 guard (seeds 0-14)

The guard in `LiveLBM.advance` looks only at the end of each world step (10 LBM sub-steps).
Here training and evaluation were rerun and the speed was taken after **every** sub-step. All
30 reruns reproduce the committed `min_delta` exactly. Maxima are over the whole field and run.

| Arm | Training max (seed) | Median over seeds | Evaluation max (seed) | Median over seeds |
|---|---|---|---|---|
| E, monopole 2.0 | 0.259 (10) | 0.157 | 0.257 (10) | 0.147 |
| F, dipole 2.0 | 0.149 (4) | 0.119 | 0.129 (4) | 0.108 |

No sub-step crossed 0.3. The monopole margin is thin: 0.259 is 86% of the guard. Sub-step
peaks can exceed what the guard sees (F: 0.149 over sub-steps, 0.142 at step ends). The guard
is a stability limit, not an accuracy guarantee: 0.259 is a Mach number of about 0.45, where
compressibility errors (order Mach squared) are no longer small. This does not cover other
seeds, drags, sizes or the walled channels.

## Is the superposition `noself` target right? Per-agent force-masked reruns (seeds 0-4)

Fluid E, 4 diagonals, genomes cached by the mechanism study (its `official` and `noself` values
are reproduced exactly here).

- **Open loop** (the exact counterpart of the target): for each agent i, a fluid driven by all
  recorded forces except i's, along the unchanged trajectories, including the nonlinearity and
  the mean-flow controller's response.
- **Closed loop:** the whole simulation rerun with agent i not forcing (16 reruns per rollout),
  scoring agent i only. Trajectories and behaviour change, including every other agent's.

| Genomes | Superposition error against open loop, mean / max | Own wake (open loop) | Alignment: superposition / open / closed / official |
|---|---|---|---|
| B-trained | 0.0047 / 0.078 | 0.083 | 0.624 / 0.627 / 0.648 / 0.695 |
| E-trained | 0.0021 / 0.024 | 0.066 | 0.575 / 0.577 / 0.609 / 0.650 |
| initial | 0.0030 / 0.032 | 0.081 | 0.059 / 0.060 / 0.047 / 0.241 |

Errors are vector differences at the agent's cell in flow units (law magnitude 0.2).

- **Open loop validates the target.** The mean error is 3-6% of the own wake (individual
  agent-steps reach 0.078, about one wake) and mean alignments differ by at most 0.003. The
  per-seed margin changes below differ by at most 0.002.
- **Closed loop is a different quantity**, and it moves the margin more:

| Margin change, fluid E minus fluid B, mean of 5 seeds | official | superposition | open | closed |
|---|---|---|---|---|
| B-trained | -0.049 | +0.017 | +0.017 | +0.064 |
| E-trained | -0.067 | -0.017 | -0.017 | +0.026 |

With 5 seeds none of these is tested for significance, and these seeds show a smaller official
loss than the 15-seed mean (-0.093 and -0.116). The direction agrees with the parent reading
(the loss goes away without the own wake). The closed-loop values do not support a remaining
cost for E-trained genomes either, but are too few to exclude one. Closed-loop scores use one
agent per rerun, so no single trajectory has every agent unable to stir its own target.

Reproduce: `measure_monopole_mechanism.py train --out CACHE`, then
`measure_lbm_live_followups.py masked --out DIR --cache CACHE --seeds 0 1 2 3 4`, and
`measure_lbm_live_followups.py speed --out DIR` (merge the per-run JSONs into `speed.json`).
Per-seed rows: `speed.json`, `masked/`.
