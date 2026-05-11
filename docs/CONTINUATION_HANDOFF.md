# Continuation Handoff — hybrid-alife

> Purpose: enable a brand-new Claude (or human) chat starting from zero context to
> continue this project without re-deriving state. Read this first, then `README.md`,
> then `docs/scientific_validation.md`.

## Pointers

- **Repo URL:** https://github.com/sebastbernal2-ship-it/hybrid-alife
- **Main branch tip before this sprint:** `0c25070bc0c29d82371a68e1e150704c5ddbded1`
  (`merge: sprint/poet-transfer-fast into final integration`)
- **This sprint branch:** `sprint/continuation-handoff-fast` (docs-only).
- **Date of handoff:** 2026-05-11.

## TL;DR project state

Hybrid artificial-life simulator in JAX. Two coupled branches (embodied GRU/MLP agents +
Avida-style instruction VM) share a 2D world of proxy microfluidic fields (curvature,
shear, lift, enrichment, concentration, metabolites). Evolution loop with tournament,
novelty archive, MAP-Elites. Scientific-depth metrics (Bedau, QD, topsim/posdis/bosdis,
Hill 1D lineage) are implemented and unit-tested. A v1 transfer matrix and a compute-
scaling harness exist as scripts but POET-style coevolution is **not** yet wired in.

## Known validation (snapshot at handoff)

- `pytest -q` → **104 tests passed in ~83 s** on CPU.
- QD-active smoke (`configs/qd_active.yaml`) at gen 5:
  - coverage `0.4375`, qd_score `77.9649`, novelty archive size `72`.
- Communication benchmark acceptance:
  - compositional protocol topsim / posdis / bosdis all `1.0`.
- Compute-scaling, budget 6, gen 5:
  - coverage `0.5`, qd_score `15.4245`, novelty archive size `36`.
- Transfer/scaling artifacts exist under `outputs/` from prior sprints.

## Architecture map

```
src/hybrid_alife/
├── agents/
│   ├── controller.py     GRU / MLP forward + decode_actions; vectorized einsum
│   ├── embodied.py       embodied lifecycle, world coupling, reproduction
│   └── avida_vm.py       instruction-genome VM, logic tasks, merit cycles
├── world/
│   ├── env.py            WorldState, terrain/fields, advection, Dean curvature
│   └── sensing.py        sixth-sense sampling, noise / blind / shuffle / delay
├── evolution/
│   ├── selection.py      tournament, reseeding
│   └── archives.py       novelty + MAP-Elites
├── experiments/
│   ├── runner.py         config loader + main run loop
│   ├── shadow.py         neutral-shadow runner (Bedau control)
│   └── transfer.py       median/IQR, bootstrap CI, Cliff's δ
├── metrics/
│   ├── core.py           per-step survival, entropy, comms, enrichment
│   ├── bedau.py          A_new, A_cum, A_p
│   ├── qd.py             QD-score, coverage, archive entropy
│   ├── communication.py  topsim / posdis / bosdis + MI channel capacity
│   ├── comm_benchmark.py synthetic comm benchmark with control protocols
│   └── lineage.py        LineageTree, Hill 1D effective lineage count
├── logging/              JSONL writer
├── replay/               deterministic pickle checkpoints
├── viz/                  matplotlib plotting helpers
└── types.py              shared dataclasses / configs

scripts/
├── run_sim.py                main runner (config → outputs/runs/<name>/)
├── generate_report.py        builds report.md + plots from a run dir
├── run_ablation_matrix.py    multi-config × multi-seed sweep + stats
├── run_comm_benchmark.py     synthetic compositionality benchmark
├── run_compute_scaling.py    budget-vs-coverage scaling harness
└── run_transfer_matrix.py    v1 transfer evaluation across configs

configs/    YAML configs (base, smoke200, 11 ablations, qd_active,
            comm_task, lineage_growth, scaling_tiny, transfer_{source,target})
tests/      9 test files, 104 tests total, all CPU-feasible
docs/       architecture, world_model, agent_branches, scientific_validation,
            qd_active, comm_benchmark, poet_transfer, lineage_growth,
            preregistration_template, CONTINUATION_HANDOFF (this file)
```

Data flow: `WorldState` is the shared substrate. Embodied agents deposit /
consume resources and metabolites; Avida organisms do I/O via the same fields.
Both branches feed into the evolution loop (selection → mutation → reseeding)
and the novelty + MAP-Elites archives. Per-step metrics stream to JSONL.

## Implemented features (what is real today)

**World / sensing**
- Proxy microfluidic fields: curvature regimes, Dean secondary flow, advection,
  drift, shear, shear-gradient, inertial-lift proxy, margination/enrichment,
  diffusing concentration waves, metabolites.
- Sensing modes: true / delayed / noisy / dropout / shuffled / blind +
  crowding-context channel from occupancy and neighbor message energy.

**Agents**
- Embodied branch: GRU controller with MLP ablation (`use_memory=false`),
  bounded continuous + discrete action vector, action-cost model, gated
  message channel.
- Avida branch: instruction-genome VM with logic-task rewards, merit-based
  CPU-cycle allocation, duplication mutation, branch coupling via metabolites.

**Evolution**
- Tournament selection, reproductive reseeding into dead slots, novelty
  archive, MAP-Elites archive with QD logging.

**Metrics & validation**
- Per-step core metrics (survival, entropy, comms usage, coordination,
  enrichment separation, mean Avida merit, logic-task count).
- Bedau-Packard activity (A_new, A_cum, A_p) with paired neutral shadow.
- QD summary (qd_score, coverage, archive + occupancy entropy).
- Compositionality (topsim, posdis, bosdis) + channel ablations + MI capacity.
- Lineage tree JSON export + Hill 1D effective lineage count.
- Transfer suite: median+IQR, bootstrap CI, Cliff's δ effect size, markdown
  summary writer. **Caveat:** v1 re-evaluates per-config metrics, not true
  trained-on-A-evaluated-on-B agent transfer.
- Compute-scaling harness (`run_compute_scaling.py`): budget × coverage curves.
- Synthetic communication benchmark with compositional / scrambled / random
  control protocols.

**Infrastructure**
- Deterministic pickle checkpoints + JSONL metric stream + auto report.
- GitHub Actions CI runs the full pytest suite on push.
- 19 YAML configs covering base, smoke200, 11 ablations, qd_active, comm_task,
  lineage_growth, scaling_tiny, transfer source/target.

## Not implemented / experimental (what is **not** real yet)

- **Full POET-style coevolutionary loop.** The transfer matrix is v1
  re-evaluation only. No environment-generator / agent-population coevolution.
- **Compositionality on the continuous emergent channel.** Currently
  discretised via argmax; numbers should be read as a lower bound.
- **Surrogate-assisted QD** and **Meta-Referential evaluation** (memo P2) —
  scaffolded only.
- **Headline numbers at scale.** All CI runs are deliberately tiny.
  Anything publishable should be re-run at ≥10 seeds with longer horizons.
