# Hybrid Artificial-Life Simulator

This repo is the first working scaffold for a hybrid artificial-life research system.

The locked direction is:

- Embodied JAX artificial-life branch with recurrent neural agents, evolutionary reproduction, and a rich 2D world.
- Avida-inspired instruction-genome branch with digital organisms, a virtual CPU, instruction execution, replication, and mutation.
- Shared proxy-physics substrate inspired by inertial microfluidics: curvature, Dean-flow proxy, shear gradients, inertial lift proxy, enrichment/margination analogs, and concentration waves.
- Shared experiment, logging, replay, and metrics framework for open-ended evolution, emergent communication, adaptation, and convention-like behavior.

This is not a CFD simulator and has no wet-lab requirement. All physics-like signals are computational proxy fields designed to create structured, uncertain sensory ecology.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python scripts/run_sim.py --config configs/base.yaml
```

## Repo layout

```text
src/hybrid_alife/
  agents/       # embodied recurrent agents and instruction-genome VM organisms
  world/        # shared 2D substrate, fields, resources, hazards, occupancy
  evolution/    # parent selection, mutation schedules, extinction handling
  experiments/  # experiment runners and config resolution
  logging/      # event logs, metric sinks, checkpoints
  metrics/      # survival, diversity, communication, enrichment, transfer metrics
  replay/       # deterministic replay records and reconstruction
  utils/        # random keys, pytrees, serialization helpers
configs/        # YAML experiment configs
experiments/    # named experiment plans
notebooks/      # exploratory analysis
tests/          # unit and smoke tests
docs/           # architecture docs
scripts/        # CLI entrypoints
outputs/        # ignored runtime outputs
```

## Current implementation status

Implemented now:

- Typed config and state dataclasses.
- Shared grid world skeleton with proxy field channels.
- Embodied agent genome/controller/action skeleton.
- Avida-style VM skeleton with instruction set and replication placeholders.
- Experiment runner that initializes state and executes a smoke simulation loop.

Next file to build first is probably `src/hybrid_alife/world/env.py`, because every branch depends on the world state transition.

