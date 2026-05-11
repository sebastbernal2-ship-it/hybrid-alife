# Agent Branches

This document explains how the embodied and Avida branches of hybrid-alife
are structured, how they interact through the shared world, and which
configuration knobs control each behavior. It targets readers who want to
extend the agent code or design ablation experiments.

The implementations live in:

- `src/hybrid_alife/agents/controller.py` — pure neural controller (GRU / MLP)
- `src/hybrid_alife/agents/embodied.py` — embodied lifecycle + world coupling
- `src/hybrid_alife/agents/avida_vm.py` — Avida-style instruction VM

## Embodied Branch

### Controller

Each embodied agent owns a copy of a small GRU-like controller. The genome
is a dict of per-individual parameter tensors so mutation is a Gaussian
perturbation on the array stack — no separate model state. The forward pass
is implemented in `controller.py::forward` as a vectorized einsum:

```
z   = sigmoid(obs · W_xz + h · W_hz + b_z)        # update gate
h~  = tanh   (obs · W_xh + h · W_hh + b_h)         # candidate
h'  = (1 - z) * h + z * h~                          # GRU mix
```

When `embodied.use_memory` is false we collapse the recurrence (`h' = h~`),
turning the controller into a stateless MLP. This is the canonical
no-memory ablation.

### Action vector

`decode_actions` turns the raw logits into a bounded action vector of
length `8 + message_size`:

| index | meaning                          | mapping                |
| ----- | -------------------------------- | ---------------------- |
| 0, 1  | continuous movement (`-1..1`)    | `tanh(raw / T)`        |
| 2     | eat gate                         | `sigmoid(raw / T)`     |
| 3     | attack gate                      | sigmoid                |
| 4     | reproduce gate                   | sigmoid                |
| 5     | terraform-resource gate          | sigmoid                |
| 6     | terraform-hazard gate            | sigmoid                |
| 7     | emit-message gate                | sigmoid                |
| 8..   | message payload (linear)         | identity               |

When `embodied.stochastic_actions` is true the six gates are Bernoulli-
sampled, which produces a genuinely discrete policy useful for entropy /
diversity ablations.

### Observation tensor

`observe_embodied` concatenates three blocks per agent:

1. **Patch**: a `(2r+1)^2` square neighborhood of every world field
   (`build_world_field`). The field order is fixed in `patch_channels`, so
   if `WorldConfig` gains new channels you only need to extend the field
   list — the controller dimensions adapt automatically.
2. **Six sixth-sense modalities** (`sample_sixth_sense`): Dean-flow
   curvature, shear gradient, margination/enrichment, inertial lift,
   concentration channels, and a crowding-context vector (neighbor count,
   local flow speed, neighbor message energy). Each can be ablated
   independently via `blind`, `true_sixth_sense`, `shuffle_sixth_sense`.
3. **Proprioception**: position, energy, age, alive flag, hidden-state
   norm, and the agent's own message (zeroed if `use_comms=False`).

If a future world extension adds a per-cell channel that the controller has
not been trained on, the patch is still the right shape because every
helper queries channel widths from `WorldConfig`. Adding new sixth-sense
modalities requires extending `sixth_sense_dim` plus a fresh genome init,
which the controller does automatically since `embodied_observation_dim`
re-derives the input size.

### Action application

`apply_embodied_actions` is the deterministic, testable update used by both
the runner and the tests:

| Effect             | Driven by                | Quantity                                      |
| ------------------ | ------------------------ | --------------------------------------------- |
| Movement           | `move_x`, `move_y`       | up to ±5% of the world per step               |
| Eat                | `eat` gate               | drains `cell_resources * 0.2 * gate`          |
| Hazard damage      | passive + `attack` gate  | `attack` halves incoming damage               |
| Hazard deposit     | `attack` gate            | `0.02 * attack` per cell (toggle via config)  |
| Terraform-resource | `terra_r` gate           | positive concentration deposit                |
| Terraform-hazard   | `terra_h` gate           | negative concentration deposit                |
| Metabolite deposit | always (eat-modulated)   | `0.01 + 0.05 * eat` per cell                  |
| Emit message       | `emit > threshold`       | hard-gated; below threshold = decay only      |
| Reproduce          | `repro` gate + threshold | handled in `apply_reproduction` next step     |

### Action cost model

Every action class has an explicit cost added to the per-step energy
balance (`EmbodiedConfig.cost_*`). The defaults are small so existing
runs remain comparable, but `cost_eat`, `cost_attack`, `cost_reproduce`,
`cost_terraform`, `cost_emit`, and `cost_move_scale` let you penalize
specific behaviors to study trade-offs.

### Reproduction

`apply_reproduction` is an asexual, deterministic operator: living agents
with energy ≥ `reproduce_energy_threshold` and `repro_gate > 0.5` are
matched 1:1 to dead slots in argsort order. The child inherits a mutated
copy of the parent's genome dict, half the initial energy, and a fresh
hidden state. Lineage IDs and depths are tracked for novelty / map-elites.

## Avida Branch

### Instruction set

| op           | id | semantics (vectorized)                                       |
| ------------ | -- | ------------------------------------------------------------ |
| NOP          | 0  | no-op                                                        |
| INC, DEC     | 1, 2 | register 0 +/- 1                                           |
| ADD, NAND    | 3, 4 | reg0 = reg0 ± reg1, or NAND(reg0, reg1)                    |
| SHIFT_L/R    | 5, 6 | bitwise shift on reg0                                      |
| LOAD_ENV     | 7  | reg0 ← quantized enrichment, reg1 ← quantized concentration  |
| EMIT         | 8  | publish reg0; if it matches a logic task, credit merit       |
| H_ALLOC      | 9  | reset copied buffer and write head                           |
| H_COPY       | 10 | copy genome[r_head] → copied[w_head] with point mutation    |
| H_DIVIDE     | 11 | spawn offspring once enough has been copied                  |
| JUMP / JZ    | 12, 13 | unconditional / conditional jump to reg0 mod genome_len  |
| SET_READ/WR  | 14, 15 | move r_head or w_head to reg0 mod max_len                |

### Logic-task scoring

`_logic_task_outputs` enumerates nine classic Avida tasks (NOT, NAND, AND,
OR_N, OR, AND_N, NOR, XOR, EQU). Each `LOAD_ENV` updates the organism's
two input registers. Each `EMIT` compares register 0 against every task
output: if it matches and the task bit isn't already set, the bit is set
in `tasks_completed` and the task reward (configurable per-task) is added
to the organism's merit. Bits are reset on division so a child organism
must re-earn its rewards.

### CPU-cycle allocation

The original Avida design rewards merit with extra CPU time. We implement
this with `_cycle_budget`:

```
budget = base + base * (merit / merit_floor) ^ merit_cycle_exponent
budget = clip(budget, 1, max_cycles_per_update)
```

`step_avida_population` always loops `max_cycles_per_update` times, but
the per-organism `cycle_mask = budget > c` gates whether each organism
participates in that cycle. This keeps the loop shape static (jit-friendly)
while still giving higher-merit organisms more compute.

### Replication and mutation operators

Replication uses `H_ALLOC` / `H_COPY` / `H_DIVIDE`. At divide time we apply
three independent stochastic operators, each bounded by `max_genome_length`
and `min_genome_length`:

- **Point mutation** during `H_COPY` (`point_mutation_prob`).
- **Insertion / deletion** at divide (`insertion_prob`, `deletion_prob`).
- **Duplication** of a random window of the copied buffer
  (`duplication_prob`), with length capped at one quarter of the copied
  region so it does not blow up the genome on a single division.

### Reseeding

When the population collapses, `reseed_avida` randomizes the dead slots'
genomes but re-injects the self-replicator template at the start so that
recovery does not depend on finding replication ab initio.

## Branch Interaction

The world is the only coupling channel. Two-way exchange happens through
metabolites and concentration:

- **Embodied → world**: every embodied agent deposits a metabolite scalar
  proportional to its eat gate, and (when attacking) deposits hazard. The
  strength is gated by `metabolite_deposit_scale` / `hazard_deposit_scale`.
- **World → Avida**: each digital organism is mapped onto an embodied
  agent's grid cell (wrap-around for population mismatches) so that
  `LOAD_ENV` reads the local enrichment and concentration that embodied
  behavior is currently shaping.
- **Avida → world**: when `metabolite_uptake > 0` digital organisms consume
  a fraction of the local metabolite field (and gain merit proportional to
  the take); when `metabolite_deposit > 0` they push concentration back
  into the world. This closes the resource loop: embodied agents seed the
  metabolite field, digital organisms harvest it, and the harvest leaves a
  concentration trail that embodied agents sense via their sixth sense.

Disable the coupling by setting `metabolite_uptake = 0` and
`metabolite_deposit = 0`, which is the ablation baseline for studies that
need the two branches to be independent.

## Ablation cheat-sheet

| Ablation                  | Knob                                     |
| ------------------------- | ---------------------------------------- |
| Blind agents              | `embodied.blind = true`                  |
| No memory (MLP)           | `embodied.use_memory = false`            |
| No communication          | `embodied.use_comms = false`             |
| Shuffled sixth sense      | `embodied.shuffle_sixth_sense = true`    |
| Stochastic policy         | `embodied.stochastic_actions = true`     |
| Branches decoupled        | `avida.metabolite_uptake = 0` (+ deposit)|
| No merit-based CPU        | `avida.merit_cycle_exponent = 0`         |
| Frozen genome             | `avida.point_mutation_prob = 0`          |

## Tests

`tests/test_agents_vm_depth.py` covers every action class, every controller
ablation, every Avida operator (`H_ALLOC` reset, copy-then-divide, EMIT
with logic-task scoring, duplication), and the merit/cycle-budget
mapping. Run with `pytest -q tests/test_agents_vm_depth.py`.
