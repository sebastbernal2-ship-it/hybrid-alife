# Continuous ecosystem: persistence and behaviour evidence

Config `configs/ecosystem.yaml`, runner `scripts/run_ecosystem.py`, 5 seeds x 80,000 steps.
No generations, selection operator, elite copies or reseeding: after 200 random founders,
individuals appear only by asexual birth from a living parent. Exploratory: the regime was
tuned by hand on seeds 0-3 (survival, turnover and how it looked) before these runs, nothing
was pre-registered, and the same seeds appear in tuning and results.

## Why the legacy setup was not enough

`legacy_probe.py` loops `step_sim` with the physics-study config (Avida off, ledger on, no
generation selection), 3 seeds x 3,000 steps (`legacy_probe.txt`). Seeds 0 and 1 stay at the
32-slot cap while births fall from about 60 to 3-4 per 500 steps (no ageing, so survivors
keep their slots). Seed 2 is left with 2 agents and no births from step 1,000.

## What is added (all opt-in, legacy defaults unchanged)

- Core: `compact_senses` (10 egocentric channels), `controller_init_scale`, `move_scale`,
  `senescence_age`/`senescence_cost`, and a `last_intake` record. The reproduction slot-0 bug
  is avoided by the existing `energy_ledger` path.
- Environment (`ecosystem.py`): food pellets in drifting patches, drifting toxic patches, and
  rot that ramps from 0 to 0.6% per step over steps 3,000-13,000 on a fixed schedule. In
  short tuning runs founders died out when rot applied from the start, so the rich start is
  part of the design.
- Body size and elongation act through eating rate, Stokes drag and metabolism.

## Persistence and turnover

Windows are 2,000 steps; ranges cover windows after step 14,000. Slot capacity is 400; the
largest population after step 14,000 was 136. Full per-window series (births, deaths by
cause, population, founder lines): `seed_*/summary.json`.

| seed | births | deaths | births/window | deaths/window | population | founder lines (step 2k / 80k) | mean generation at 80k |
|---|---|---|---|---|---|---|---|
| 0 | 15687 | 15789 | 297-458 | 289-455 | 38-119 | 1 / 1 | 270 |
| 1 | 18152 | 18248 | 333-521 | 339-524 | 40-136 | 4 / 1 | 337 |
| 2 | 18503 | 18616 | 342-642 | 349-636 | 44-133 | 1 / 1 | 291 |
| 3 | 18298 | 18381 | 397-504 | 385-510 | 47-129 | 1 / 1 | 288 |
| 4 | 18041 | 18131 | 378-534 | 354-556 | 49-124 | 1 / 1 | 301 |

Every seed first crashes from 200 founders and ends with a single founder line (four seeds by
step 2,000). Viewer family colours are therefore sub-families of that line.

## Common garden

100 genomes per group, no reproduction, 300 steps, 8 fresh worlds (1000-1007) shared by all
groups (`seed_*/garden.json`). Evolved genomes are sampled from those alive at step 80,000.
Trial worlds start food-rich with no rot, so they are easier than the late-run world.
"Mutation only" applies the median generation depth of mutations without selection (not a
per-lineage replay). Worlds are repeated measures; the independent replicate is the training
seed (n = 5). No significance test was run.

### Food energy eaten per genome

| seed | founders | mutation only | founder brain, evolved body | evolved | evolved, food cut | evolved, senses cut |
|---|---|---|---|---|---|---|
| 0 | 22 | 22 | 34 | 118 | 54 | 56 |
| 1 | 23 | 26 | 37 | 101 | 72 | 75 |
| 2 | 22 | 22 | 37 | 122 | 57 | 57 |
| 3 | 22 | 23 | 34 | 129 | 59 | 50 |
| 4 | 22 | 25 | 33 | 130 | 53 | 50 |

### Share alive at step 300

| seed | founders | mutation only | founder brain, evolved body | evolved | evolved, food cut | evolved, senses cut |
|---|---|---|---|---|---|---|
| 0 | 0.17 | 0.15 | 0.29 | 0.87 | 0.51 | 0.45 |
| 1 | 0.17 | 0.19 | 0.32 | 0.81 | 0.63 | 0.67 |
| 2 | 0.15 | 0.15 | 0.30 | 0.83 | 0.54 | 0.45 |
| 3 | 0.16 | 0.16 | 0.30 | 0.93 | 0.64 | 0.41 |
| 4 | 0.17 | 0.16 | 0.28 | 0.85 | 0.53 | 0.37 |

### Cosine between actual displacement and sensed food direction

| seed | founders | mutation only | founder brain, evolved body | evolved | evolved, food cut | evolved, senses cut |
|---|---|---|---|---|---|---|
| 0 | -0.01 | -0.00 | -0.01 | 0.50 | 0.04 | 0.07 |
| 1 | -0.01 | -0.00 | -0.01 | 0.29 | 0.03 | 0.04 |
| 2 | 0.01 | -0.01 | 0.00 | 0.53 | 0.05 | 0.08 |
| 3 | -0.00 | -0.01 | 0.00 | 0.63 | 0.07 | 0.10 |
| 4 | -0.02 | -0.00 | -0.01 | 0.61 | 0.05 | 0.08 |

### Share of agent-steps with a toxin underfoot

| seed | founders | mutation only | founder brain, evolved body | evolved | evolved, food cut | evolved, senses cut |
|---|---|---|---|---|---|---|
| 0 | 0.035 | 0.033 | 0.042 | 0.032 | 0.013 | 0.033 |
| 1 | 0.036 | 0.036 | 0.033 | 0.023 | 0.024 | 0.035 |
| 2 | 0.037 | 0.033 | 0.038 | 0.034 | 0.036 | 0.033 |
| 3 | 0.034 | 0.037 | 0.032 | 0.027 | 0.024 | 0.032 |
| 4 | 0.035 | 0.036 | 0.033 | 0.027 | 0.014 | 0.032 |

In all 5 seeds evolved genomes eat several times more and survive far more often than
founders and the mutation-only control, and they steer toward food. Cutting the three food
channels removes the steering and much of the gain. Evolved body genes alone give a small
part of the gain.

## Limits

- Cutting senses leaves evolved genomes well above founders, so a large part of the gain is
  not food sensing (movement habits, eating gate, body).
- No clear evidence of learned toxin avoidance (table above is mixed, and the lowest values
  appear with food sensing cut).
- Body genes run into their clip limits (size 1.8, elongation 0.7 or 1.4) in most seeds, so
  late populations look alike. Larger is favoured by this rule set.
- Attacks become common late in several seeds. "Died while attacked" only means the agent
  was a target in its final step, not that the attack was the sole cause.
- One hand-built world. Nothing here shows open-ended evolution. No collisions (bodies
  overlap), asexual reproduction, no fluid.

## Reproduce

```
make -C native
export PYTHONPATH=src JAX_PLATFORMS=cpu OMP_NUM_THREADS=1
python scripts/run_ecosystem.py run --seed 3 --steps 80000 --checkpoint 77000 --out outputs/ecosystem
python scripts/run_ecosystem.py garden --run outputs/ecosystem/seed_3
python scripts/run_ecosystem.py record --run outputs/ecosystem/seed_3 --checkpoint 77000 \
  --steps 3000 --out outputs/ecosystem/replay.html
```

About 25 minutes per run with five in parallel. Checkpoints are pickles; load only your own.

## Viewer

Standalone HTML (Canvas 2D, no network). Headless Windows Chrome (D3D11, RTX 5080 Laptop
GPU, 1302x832 world canvas in a 1920x1080 window): 240 fps mean, p95 4.3 ms, worst frame
8.2 ms in three scenarios (overview; 4x follow with inspector; 80 steps/s with inspector).
240 is the headless pacing cap. Integrated GPUs and headful windows were not measured.

Drawn but not simulated: membranes, eyespots, cilia, tail beat, heading (from movement),
motion between steps, event animations, drawn body scale (bodies are drawn larger than the
interaction distances). Family hue is not a gene.
