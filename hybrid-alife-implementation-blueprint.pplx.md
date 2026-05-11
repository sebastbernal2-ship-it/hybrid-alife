# Hybrid Artificial-Life Implementation Blueprint

## Research anchors

The embodied branch follows the JaxLife pattern of JAX-native embodied neural agents acting in a rich world, where evolved recurrent controllers can support rudimentary communication, agriculture, and tool use ([JaxLife arXiv](https://arxiv.org/abs/2409.00853)). The digital-organism branch follows the Avida pattern: self-replicating instruction genomes execute on a virtual CPU, mutate during replication, and experience differential fitness through competition for memory/space and CPU time ([Nature Scientific Data](https://www.nature.com/articles/s41597-023-02514-3)). The proxy field model is inspired by inertial microfluidics, where curvature-induced Dean flow, shear-gradient lift, wall/lift interactions, and secondary-flow drag jointly shape focusing, migration, and separation behavior ([Inertial Focusing in Microfluidics](https://pmc.ncbi.nlm.nih.gov/articles/PMC4467210/)).

## System spec

### Major modules

| Module | Path | Responsibility | V1 status |
| --- | --- | --- | --- |
| Config | `configs/*.yaml`, `experiments/runner.py` | Resolve experiment parameters into typed dataclasses. | Implement now |
| Types | `src/hybrid_alife/types.py` | Shared dataclasses for world, populations, configs, and simulation state. | Implement now |
| World | `src/hybrid_alife/world/env.py` | Owns grid fields, resources, hazards, proxy flow, curvature, shear, enrichment, lift, concentration waves, and occupancy. | Implement now |
| Embodied agents | `src/hybrid_alife/agents/embodied.py` | Neural recurrent agents, observations, actions, genome mutation, reproduction hooks. | Implement now |
| Avida VM | `src/hybrid_alife/agents/avida_vm.py` | Instruction set, virtual CPU state, VM execution, replication hooks, mutation. | Implement now |
| Evolution | `src/hybrid_alife/evolution/` | Parent selection, offspring generation, mutation schedules, extinction recovery. | V1 skeleton |
| Metrics | `src/hybrid_alife/metrics/` | Survival, reproduction, diversity, novelty, communication, ritual/convention, robustness, transfer, enrichment metrics. | V1 skeleton |
| Logging | `src/hybrid_alife/logging/` | Metrics sink, event stream, checkpoints. | Future extension |
| Replay | `src/hybrid_alife/replay/` | Deterministic reconstruction from seeds, configs, and event deltas. | Future extension |
| Scripts | `scripts/run_sim.py` | CLI entrypoints. | Implement now |
| Tests | `tests/` | Unit and smoke tests. | Implement now |

### Interfaces between modules

`ExperimentConfig` is the only public config object consumed by the runner. `WorldState` is the shared substrate consumed by both branches. `EmbodiedPopulationState` and `AvidaPopulationState` are branch-local population states. `SimState` binds all state plus RNG, generation, step, and last metric dictionary.

The runner calls `initialize_world`, `initialize_embodied_population`, and `initialize_avida_population` at startup. Each step calls `step_world`, then branch-local stepping functions. Evolution is applied at generation boundaries after `steps_per_generation` environment steps.

### Data flow

```text
YAML config
  -> typed ExperimentConfig
  -> initialize SimState
  -> step_world updates passive substrate
  -> embodied observe -> recurrent controller -> action effects
  -> Avida VM cycles -> emit/consume world signals -> replication effects
  -> metrics/logging/replay
  -> generation-boundary evolution
```

### Execution loop

```python
for generation in range(cfg.evolution.generations):
    for t in range(cfg.evolution.steps_per_generation):
        state = step_world(state, cfg.world)
        state = step_embodied_branch(state, cfg.embodied)
        state = step_avida_branch(state, cfg.avida)
        if t % metrics_every == 0:
            log_metrics(state)
    state = evolutionary_update(state, cfg.evolution)
    checkpoint_and_replay(state)
```

### Branch separation

The embodied branch owns continuous positions, energy, recurrent hidden state, messages, and neural genomes. The Avida branch owns discrete genome arrays, instruction pointers, registers, memory, copy buffers, merit, and replication state. The branches do not share genome formats, reproduction mechanics, or controller implementations.

The shared world is the only coupling surface in v1. Embodied agents deposit messages and concentration signals, consume resources, and change occupancy. Avida organisms read environmental inputs, emit digital metabolites/messages, compete for lattice slots, and eventually modify local concentration/resource channels.

### Shared experiment, logging, replay, and metrics framework

Every run is identified by `(run_name, seed, config_hash)`. V1 writes scalar metrics and checkpoints. V2 adds event logs for births, deaths, attacks, emissions, reproductions, VM task completions, and terrain edits. Replay should reconstruct a run from config, seed, branch event deltas, and checkpoint snapshots.

## Repo scaffold

```text
hybrid-alife/
  README.md
  pyproject.toml
  configs/
    base.yaml
  experiments/
    smoke_hybrid_v1.md
  notebooks/
    .gitkeep
  tests/
    test_smoke.py
  docs/
    architecture.md
  scripts/
    run_sim.py
  outputs/
    .gitkeep
    runs/
    replays/
    checkpoints/
    metrics/
  src/hybrid_alife/
    __init__.py
    types.py
    world/
      __init__.py
      env.py
    agents/
      __init__.py
      embodied.py
      avida_vm.py
    evolution/
      __init__.py
      selection.py
    experiments/
      __init__.py
      runner.py
    logging/
      __init__.py
    metrics/
      __init__.py
      core.py
    replay/
      __init__.py
    utils/
      __init__.py
```

## World model

### Recommended choice

Implement now: a toroidal 2D lattice with continuous agent positions projected onto grid cells. This keeps JAX vectorization simple while allowing smooth fields and later continuous interpolation.

Later alternative: fully continuous 2D space with spatial hashing. Defer until baseline evolution works.

### Concrete data structures

```python
WorldState(
    terrain:      Float32[H, W, 4],   # base cost, fertility, permeability, information bit/proxy
    resources:    Float32[H, W, R],   # food/metabolite/resource channels
    hazards:      Float32[H, W, Z],   # toxicity/predation/friction channels
    flow:         Float32[H, W, 2],   # Dean-like secondary velocity proxy
    curvature:    Float32[H, W, 1],   # signed channel/world curvature proxy
    shear:        Float32[H, W, 2],   # local gradient of flow components
    enrichment:   Float32[H, W, 1],   # margination/enrichment affordance
    lift:         Float32[H, W, 2],   # inertial lift proxy vector
    concentration:Float32[H, W, C],   # communication/metabolite waves
    occupancy:    Int32[H, W],        # -1 empty, otherwise agent/organism index
)
```

### Channel definitions

- `terrain[..., 0]`: movement/action cost multiplier.
- `terrain[..., 1]`: resource regeneration rate.
- `terrain[..., 2]`: permeability/diffusion multiplier.
- `terrain[..., 3]`: persistent local information bit/proxy marker.
- `resources[..., r]`: consumable energy or digital-resource fields.
- `hazards[..., z]`: damage, action penalty, or mutation-pressure field.
- `flow[..., :]`: abstract secondary flow that advects concentration and biases movement.
- `curvature[..., 0]`: signed local curvature, used to compute Dean-like flow.
- `shear[..., :]`: finite-difference gradient of flow.
- `enrichment[..., 0]`: high where resources, flow, and lift create separation-like affordance.
- `lift[..., :]`: local cross-stream migration bias.
- `concentration[..., :]`: communication waves and emitted digital metabolites.

### Occupancy and interactions

Implement now: one occupant per grid cell for hard collisions, with embodied continuous positions mapped by `floor(pos * [W, H])`. Avida organisms occupy grid slots directly. If both branches target the same cell, resolve in this order: hazards/death, reproduction/birth, movement, emission, resource consumption.

Future extension: allow layered occupancy with soft density fields and local crowding pressure.

## Agent designs

### Embodied branch

Observation tensor:

```python
obs_i = concat(
    local_patch(world_channels, radius=r),      # [(2r+1), (2r+1), Cw] flattened
    nearest_agents_features,                    # later: K nearest
    self_features,                              # energy, age, pos, hidden norm, alive
    received_message_features,                  # local concentration + neighbor messages
)
```

V1 world channels are terrain, resources, hazards, flow, curvature, shear, enrichment, lift, and concentration. With radius 5, this is a small patch and can be gathered with vectorized indexing.

Action space:

| Action | Type | Effect |
| --- | --- | --- |
| `MOVE_X`, `MOVE_Y` | continuous | Delta position, biased by terrain and lift. |
| `EAT` | continuous gate | Consume local resource into energy. |
| `ATTACK` | continuous gate | Damage or steal energy from local neighbor. |
| `REPRODUCE` | continuous gate | Spawn mutated offspring if energy threshold passes. |
| `TERRAFORM_RESOURCE` | continuous | Modify resource regen or resource channel. |
| `TERRAFORM_HAZARD` | continuous | Modify hazard or permeability channel. |
| `EMIT_MESSAGE` | continuous vector | Deposit `message_size` signal into concentration/message field. |

Controller architecture:

Implement now: per-agent recurrent controller with `h_t = tanh(obs @ W_obs + h_{t-1} @ W_h + b_h)` and actions `a_t = h_t @ W_act + b_act`. Later replace with Flax module using CNN patch encoder, entity attention, and GRU/LSTM.

Genome representation:

```python
genome = {
    "w_obs": Float32[N, obs_dim, hidden],
    "w_h": Float32[N, hidden, hidden],
    "b_h": Float32[N, hidden],
    "w_act": Float32[N, hidden, action_dim],
    "b_act": Float32[N, action_dim],
}
```

Mutation rules:

- Implement now: elementwise Bernoulli mask plus Gaussian perturbation.
- Later: structured mutation by layer, rank-one perturbations, sparsity mutations, activation/noise genes, plasticity genes.

Reproduction rules:

- Asexual reproduction.
- Parent must be alive and exceed `reproduce_energy_threshold`.
- Child inherits parent genome with mutation.
- Parent pays `reproduction_energy_cost`.
- Child starts near parent with `initial_energy`.
- If target cell occupied, use local empty-neighbor search or fail.

Lifecycle and energy:

- Basal metabolic cost every step.
- Movement, attack, terraforming, emission, and reproduction have action-scaled costs.
- Resource consumption increases energy.
- Hazard exposure reduces energy or increases mutation pressure.
- Death if energy <= 0 or age exceeds optional max age.

### Avida-style branch

Instruction set:

| Op | Semantics |
| --- | --- |
| `NOP` | No operation. |
| `INC`, `DEC` | Mutate active register. |
| `ADD` | Add register 1 into register 0. |
| `NAND` | Boolean NAND primitive. |
| `SHIFT_L`, `SHIFT_R` | Bit shifts. |
| `LOAD_ENV` | Read local enrichment/concentration/resource input. |
| `EMIT` | Write local concentration/metabolite signal and gain task credit. |
| `H_ALLOC` | Allocate/clear copy buffer. |
| `H_COPY` | Copy instruction from read head to write head with mutation chance. |
| `H_DIVIDE` | Birth offspring if copied genome is viable. |
| `JUMP`, `JUMP_IF_ZERO` | Control flow. |
| `SET_READ`, `SET_WRITE` | Manipulate genome copy heads. |

Genome representation:

```python
genomes: Int32[N, max_genome_length]
genome_lengths: Int32[N]
```

Virtual CPU state:

```python
registers: Int32[N, R]
memory: Int32[N, M]
ip: Int32[N]
read_head: Int32[N]
write_head: Int32[N]
copied: Int32[N, max_genome_length]
merit: Float32[N]
```

Memory model:

- Genome memory is circular over `genome_lengths[i]`.
- Copy buffer is fixed-capacity and becomes child genome on `H_DIVIDE`.
- Registers are integer words.
- Local environmental input is discretized from enrichment/resource/concentration channels.

Replication semantics:

1. `H_ALLOC` clears copy buffer and resets write head.
2. `H_COPY` copies current read-head instruction to copy buffer with point mutation chance.
3. Heads advance.
4. `H_DIVIDE` checks copied length and minimum viability.
5. If viable, offspring is placed into neighboring or randomly selected grid cell, replacing occupant under configured policy.

Mutation operators:

- Point mutation on copy.
- Insertion on divide.
- Deletion on divide.
- Later: duplication, inversion, instruction-set expansion, environment-dependent mutation pressure.

Fitness and selection logic:

Selection is implicit through replication speed, survival, and competition for occupancy/CPU cycles. Merit increases from task completions such as correct environmental transformations, concentration-wave responses, or Boolean tasks over sensed inputs. Higher merit receives more VM cycles in later versions.

## Sixth-sense modalities

| Modality | Latent property | Computation | Noise/uncertainty | Agent input |
| --- | --- | --- | --- | --- |
| Dean-flow curvature detection | Signed curvature causing secondary circulation | `curvature = signed_map(x,y)`, `flow = [-curv*y, curv*x]` | Add Gaussian field noise and optional temporal drift | Local curvature patch, local flow vector, local flow divergence/curl later |
| Shear-gradient sensing | Local velocity-gradient asymmetry | Central differences of flow: `dfx/dy`, `dfy/dx` | Observation noise proportional to local hazard or turbulence | Local shear vector and magnitude |
| Margination/enrichment cue | Separation-like resource/hazard/flow affordance | `tanh(sum(resources)-sum(hazards)+0.25*||lift||)` | Smoothed over local patch; noisy if concentration high | Local enrichment scalar patch |
| Inertial lift proxy | Cross-stream migration bias | `lift = perp(shear) * ||flow|| * (1+abs(curvature)) * (0.1+||shear||)` | Noise increases with flow magnitude | Local lift vector and projected move bias |
| Concentration-wave detection | Local emitted signal and metabolite structure | Diffuse-decay stencil over concentration channels | Diffusion blur, decay, sensor Gaussian noise | Local concentration channels and temporal delta later |
| Local multi-agent hemodynamic context | Crowding-pressure analog and collective flow disturbance | Local occupancy density, message density, velocity alignment later | Occlusion and aliasing from limited radius | Local density, neighbor messages, flow disturbance channels |

Implement all six in v1 as fields or observations. Stage only advanced temporal derivatives, curl/divergence, and agent-induced flow disturbance for v2.

## Evolution and learning loop

Outer loop: evolution over generations. Within-lifetime adaptation: recurrent hidden state. Optional plasticity is deferred.

Population update:

- Embodied: environment-level births during lifetime plus generation-boundary culling/selection if population falls below target.
- Avida: VM-level births from `H_DIVIDE`; generation boundary used for metrics and optional resampling.

Parent selection:

- Implement now: tournament selection using fitness proxy `energy + reproduction_count + survival_bonus`.
- Avida: implicit replication first; tournament fallback only for reseeding after extinction.

Offspring generation:

- Copy parent genome.
- Apply branch-specific mutation.
- Assign child lineage id and parent id.
- Reset child age, hidden state/registers, and energy/merit.

Mutation schedule:

- Start fixed mutation rates.
- Later add annealed stress response: hazards and low enrichment increase mutation probability.

Extinction logic:

- If alive count below `extinction_min_population`, reseed from elite archive if available.
- If no archive, random restart the extinct branch only.
- Never reset world unless both branches collapse and no replay objective is active.

Novelty/MAP-Elites extension:

- V1: log descriptors only.
- V2: novelty archive using behavior vectors.
- V3: MAP-Elites over descriptors such as mobility, emission entropy, enrichment preference, cooperation index, replication style, and hazard tolerance.

## Metrics and validation

### Core metrics

| Metric family | Concrete metrics |
| --- | --- |
| Survival | alive count, mean lifespan, age distribution, extinction frequency |
| Reproductive success | births per generation, offspring survival, reproduction latency |
| Lineage depth | max lineage depth, active lineage count, lineage turnover |
| Behavioral diversity | action entropy, trajectory diversity, resource-use diversity |
| Exploratory novelty | distance in behavior-descriptor space, new-state visitation |
| Communication emergence | message entropy, mutual information between messages and future actions, ablation delta after zeroing messages |
| Convention/ritual proxies | repeated low-immediate-reward action sequences, synchronized emissions, stable group-specific action motifs |
| Robustness under ablation | fitness under no-flow, no-shear, no-enrichment, no-message, no-hazard variants |
| Transfer | performance after moving elite lineages across seeds, maps, curvature regimes |
| Enrichment analog metrics | separation index, enrichment preference, lift-aligned motion, concentration band formation |

### Minimum first experiments

1. Smoke initialization: world plus both branches can step for 10 generations.
2. Embodied-only survival: verify nonzero resource consumption and energy lifecycle.
3. Avida-only replication: verify `H_COPY` and `H_DIVIDE` generate viable offspring.
4. Hybrid coexistence: both branches share concentration/resource fields without shape errors.
5. Sixth-sense ablation: compare full fields vs no curvature/shear/enrichment/lift.
6. Transfer seed test: replay top embodied genomes on three new maps.

## Implementation roadmap

### First day

- Create repo scaffold.
- Implement typed configs, world initialization, proxy fields, runner, smoke tests.
- Implement placeholder embodied controller and Avida VM cycle.
- Run one smoke simulation.

### First three days

- Implement local patch observation gathering.
- Implement embodied action effects: move, eat, emit, reproduce.
- Implement occupancy updates.
- Implement Avida `H_ALLOC`, `H_COPY`, `H_DIVIDE`.
- Add scalar metrics CSV/JSONL logging.

### First week

- Add generation-boundary evolution and lineage tracking.
- Add replay checkpoints.
- Add ablation configs.
- Add first diversity and communication metrics.
- Build visualization notebook for field maps and trajectories.

### First two weeks

- Replace placeholder recurrent controller with Flax GRU/LSTM pytree genome.
- Vectorize/jit critical step functions.
- Add novelty descriptors and archive.
- Add transfer experiments across curvature regimes.
- Add branch interaction through concentration/metabolite channels.

### First month

- Add MAP-Elites.
- Add richer VM task rewards and resource competition.
- Add convention/ritual metric suite.
- Add long-run experiment management.
- Add report generation for experiment summaries.

## First coding tasks

1. Run `python scripts/run_sim.py --config configs/base.yaml`.
2. Fix any dependency/import issues.
3. Build `world/env.py` action-effect interface: `apply_embodied_actions`.
4. Implement position-to-cell and toroidal indexing helpers.
5. Implement embodied local patch observations.
6. Implement movement and occupancy resolution.
7. Implement eating and energy update.
8. Implement message emission into concentration channels.
9. Implement reproduction and genome mutation for embodied agents.
10. Implement Avida `H_ALLOC`.
11. Implement Avida `H_COPY` with point mutation.
12. Implement Avida `H_DIVIDE` viability and placement.
13. Add JSONL metrics writer.
14. Add smoke tests for each action and VM replication.
15. Add first ablation configs.

## Initial file contents

The starter implementation has been generated in this repo. The files to inspect first are:

- `README.md`
- `pyproject.toml`
- `configs/base.yaml`
- `scripts/run_sim.py`
- `src/hybrid_alife/types.py`
- `src/hybrid_alife/world/env.py`
- `src/hybrid_alife/agents/embodied.py`
- `src/hybrid_alife/agents/avida_vm.py`
- `src/hybrid_alife/experiments/runner.py`

