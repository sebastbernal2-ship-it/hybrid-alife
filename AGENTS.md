# Project agent memory

This file is the project's committed home for project-intrinsic agent knowledge: build, test, release, architecture, and sharp-edge notes that should travel with the code.

## Start here

- Research question, terms, and claim boundary: `docs/RESEARCH_ALIGNMENT.md`
- Prioritized next directions and accumulated lessons: `docs/GRAVITY_ROADMAP.md`
- Current findings and reproduction: `docs/results/gravity_study_v1/README.md`
- Evidence contract for the wider causal-inquiry program: `docs/SELECTIVE_CAUSAL_INQUIRY_PROTOCOL.md`
- Evidence tiers and how to read a metric: `docs/SCIENTIFIC_INTERPRETATION_GUIDE.md`
- Closure rules: `docs/FINALIZATION_GATE.md`
- Opt-in physics and native extensions (default off), flags and sharp edges: `docs/extensions/README.md`

## Commands

- Full suite: `venv/bin/python -m pytest -q -n 4 --dist worksteal` after `make -C native` (about 5 minutes on 4 cores, about 14 serial; several tests train small JAX models)
- Native sanitizers (ASan+UBSan, also a CI job): `make sanitize-build sanitize-test PYTHON=venv/bin/python`, then `make -B -C native` to restore the normal build.
- Lint: `venv/bin/ruff check src tests`
- Gravity study replication: `JAX_PLATFORMS=cpu venv/bin/python scripts/run_gravity_study.py --seeds 0 1 2 3 4 --output docs/results/gravity_study_v1/summary.json`
- Set `JAX_PLATFORMS=cpu` for local runs; CI runs both Python 3.11 and 3.12.
- Native C++ kernels (need g++ and make): `make -C native`; also built on first import, and a failed build raises `RuntimeError`.

## Sharp edges

- CI has a 35-minute job ceiling. Shared-runner speed can vary by about 2x between identical runs, so judge CI timing over several runs. Keep new tests fast and put full-size training runs in `scripts/`, not the default suite.
- Native versus JAX parity: integer and boolean state must match exactly; float state uses a tolerance (`rtol=1e-5, atol=1e-6`; accumulated energy totals `rtol=1e-4`) because XLA and the C++ build differ by about 1 ulp across CPUs. See `_same` in `tests/test_avida_native.py`.
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
