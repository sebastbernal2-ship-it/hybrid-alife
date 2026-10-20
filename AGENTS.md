# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Start here

- Research question, terms, and claim boundary: `docs/RESEARCH_ALIGNMENT.md`
- Prioritized next directions and accumulated lessons: `docs/GRAVITY_ROADMAP.md`
- Current findings and reproduction: `docs/results/gravity_study_v1/README.md`
- Evidence contract for the wider causal-inquiry program: `docs/SELECTIVE_CAUSAL_INQUIRY_PROTOCOL.md`
- Evidence tiers and how to read a metric: `docs/SCIENTIFIC_INTERPRETATION_GUIDE.md`
- Closure rules: `docs/FINALIZATION_GATE.md`

## Commands

- Full suite: `venv/bin/python -m pytest -q` (about 14 minutes; several tests train small JAX models)
- Lint: `venv/bin/ruff check src tests`
- Gravity study replication: `JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json`
- Set `JAX_PLATFORMS=cpu` for local runs; CI runs both Python 3.11 and 3.12.

## Sharp edges

- CI has a 35-minute job ceiling. The suite takes about 14 minutes. Keep new tests fast and put full-size training runs in `scripts/`, not the default suite.
- Never report a mean across gravity directions as the primary result. Make the minimum across held-out directions the gate, because a mean hides a single-direction collapse.
- Never average signed alignment over antipodal directions; it cancels to zero.
- Avoid reward designs with an isolated optimum. They look like learner failures and are not.
- Prefer continuous actions over a small discrete set when the direction is the thing being learned, to avoid a mode-versus-mean failure.
- Avida must stay disabled in causal traces; it confounds world outcomes.
- Keep `gravity_vector`, world flow, and other evaluator-only labels out of agent observations. `agent_observation` versus `evaluator_labels` is the boundary that keeps results honest.

## Maintaining this file

Keep this file for knowledge useful to almost every future agent session in this project.
Do not repeat what the codebase already shows; point to the authoritative file or command instead.
Prefer rewriting or pruning existing entries over appending new ones.
When updating this file, preserve this bar for all agents and keep entries concise.
