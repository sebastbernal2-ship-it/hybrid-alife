# Continuation Handoff — hybrid-alife

> Purpose: enable a brand-new Claude (or human) chat starting from zero context to
> continue this project without re-deriving state. Read this first, then `README.md`,
> then `docs/scientific_validation.md`. Before launching any multi-hour campaign,
> also read [`LONG_RUN_OPERATOR_RUNBOOK.md`](LONG_RUN_OPERATOR_RUNBOOK.md) — it
> covers preflight, env vars, sessions, resume/force policy, recovery, and the
> exact recipes for 20-min / 2-hour / overnight runs.

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

## Open scientific risks

1. **Proxy fields are stylized, not physical.** The "sixth sense" channels are
   inductive bias, not microfluidic ground truth. The `ablation_uniform_field`
   config exists to show they are doing measurable work — that comparison
   *must* accompany any claim about microfluidic-style ecology.
2. **Compositionality scores are upper-bounded by quantisation.** topsim /
   posdis / bosdis on argmax-discretised continuous messages can both
   over- and under-state structure. Treat 1.0 on the synthetic benchmark as
   tool validation, not emergent-language evidence.
3. **Transfer "matrix" is not transfer learning.** v1 re-evaluates per-config
   end-of-run metrics. A claim about generalisation requires re-running
   policies trained on env A inside env B with the same controller weights.
4. **CI runs are tiny.** Headline numbers (coverage, qd_score, novelty
   archive size) come from gen-5 smoke runs. Variance across seeds at this
   scale is large; publishable numbers need ≥10 seeds and longer horizons.
5. **Bedau A_new / A_cum / A_p depend on a paired neutral shadow.** If the
   shadow runner drifts from the live config (different sensing mode,
   different mutation rate), Bedau ratios become meaningless. Always run
   shadow with `--include-shadow` in the ablation matrix.
6. **Language guardrails.** `docs/scientific_validation.md` §7 commits to
   labelling, not biology / linguistics. Future PRs should not weaken this
   without an explicit memo update.
7. **Reseeding hides extinction.** Embodied alive-fraction stays at 1.0
   because reseeding refills dead slots. Real selection pressure has to be
   measured via lineage-depth and Hill 1D, not survival fraction.

## Next actions

### Next 20 minutes

- `pytest -q` to confirm the 104-test suite still passes locally.
- `python scripts/run_sim.py --config configs/base.yaml` (2-gen smoke) to
  confirm runner + JSONL writer are healthy end-to-end.
- Skim `outputs/` for prior sprint artifacts (transfer / scaling); confirm
  they are still readable by `generate_report.py`.

### Next 2 hours

For anything that will exceed ~10 minutes, follow
[`LONG_RUN_OPERATOR_RUNBOOK.md`](LONG_RUN_OPERATOR_RUNBOOK.md) — preflight,
tmux pattern, `--resume` policy, and the 2-hour recipe in §10b.

- Run the full smoke + ablation matrix:
  ```bash
  python scripts/run_ablation_matrix.py \
      --configs configs/smoke200.yaml configs/ablation_no_comms.yaml \
                configs/ablation_static_world.yaml configs/ablation_uniform_field.yaml \
      --seeds 3 --include-shadow \
      --out-dir outputs/ablation_matrix
  ```
- Re-run `python scripts/run_comm_benchmark.py` and confirm the compositional
  control still hits topsim/posdis/bosdis = 1.0.
- Re-run `python scripts/run_compute_scaling.py` at budgets [6, 12, 24]
  and update the scaling curve in `docs/poet_transfer.md`.
- Pick **one** experimental gap from the list above and write a short
  design memo in `docs/` before coding.

### Next 1 day

- Implement v2 transfer: load trained policies from run A's checkpoint and
  evaluate them inside config B without further mutation. Wire into
  `scripts/run_transfer_matrix.py` behind a `--mode {reeval,transfer}` flag.
- Stand up surrogate-assisted QD: a small regression head over
  behaviour-descriptor → fitness, used to gate which mutations get a full
  rollout. Acceptance: ≥30% reduction in rollouts to reach the current
  smoke200 coverage at matched seeds.
- Add a continuous-channel compositionality estimator that does not require
  discretisation (e.g. kernel topsim) and report both alongside the
  argmax-discretised baseline.
- Begin the POET loop scaffold: environment-generator population, minimal
  criterion (using the preregistered descriptor from
  `docs/preregistration_template.md`), and an environment-archive that
  parallels the MAP-Elites agent archive.

## Exact commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests (expect 104 passing, ~83 s on CPU)
pytest -q

# Lint
ruff check src tests

# Smokes
python scripts/run_sim.py --config configs/base.yaml          # 2 gens
python scripts/run_sim.py --config configs/smoke200.yaml      # 200 gens, ~110 s
python scripts/generate_report.py outputs/runs/smoke200

# Targeted experiments
python scripts/run_sim.py --config configs/qd_active.yaml
python scripts/run_sim.py --config configs/comm_task.yaml
python scripts/run_sim.py --config configs/lineage_growth.yaml
python scripts/run_sim.py --config configs/scaling_tiny.yaml

# Benchmarks / harnesses
python scripts/run_comm_benchmark.py
python scripts/run_compute_scaling.py
python scripts/run_transfer_matrix.py \
    --source configs/transfer_source.yaml \
    --target configs/transfer_target_uniform.yaml

# Full ablation matrix with stats
python scripts/run_ablation_matrix.py \
    --configs configs/smoke200.yaml configs/ablation_no_comms.yaml \
              configs/ablation_static_world.yaml configs/ablation_uniform_field.yaml \
    --seeds 3 --include-shadow \
    --out-dir outputs/ablation_matrix

# Neutral-shadow run only (Bedau control)
python -c "from hybrid_alife.experiments.runner import load_config; \
           from hybrid_alife.experiments.shadow import run_shadow; \
           run_shadow(load_config('configs/smoke200.yaml'))"
```

## Branch strategy

- `main` is the integration branch. Each completed sprint lands as a merge
  commit; see `git log --oneline main` for the chronology.
- Sprint branches use the pattern `sprint/<topic>-fast` (fast = time-boxed
  CPU-feasible). Active branches at handoff (already merged into main):
  - `sprint/lineage-growth-fast` → births + depth tracking
  - `sprint/qd-active-fast`      → active MAP-Elites + per-gen QD logging
  - `sprint/comm-task-fast`      → synthetic communication benchmark
  - `sprint/poet-transfer-fast`  → v1 transfer matrix + compute-scaling
  - `sprint/continuation-handoff-fast` → **this branch** (docs-only)
- Open a PR from each sprint branch back into `main`. CI must be green
  (full 104-test pytest) before merging. Do **not** force-push to `main`.
- Never amend merged commits; create new sprint branches off `main` for
  follow-ups.

## How to recover work from sprint branches

```bash
# List all sprint branches, local + remote
git branch -a | grep sprint/

# Inspect a sprint without checking out
git log --oneline main..origin/sprint/<topic>-fast
git diff main...origin/sprint/<topic>-fast -- <path>

# Resume an unmerged sprint
git fetch origin
git checkout -b sprint/<topic>-fast origin/sprint/<topic>-fast
# … iterate, push, open PR …

# If a sprint branch was deleted on the remote but exists locally
git push -u origin sprint/<topic>-fast

# If everything local is lost: clone fresh and recover from origin
gh repo clone sebastbernal2-ship-it/hybrid-alife
cd hybrid-alife
git fetch --all
git branch -r | grep sprint/        # list remote sprints
```

The pre-sprint integration point is the merge commit
`0c25070bc0c29d82371a68e1e150704c5ddbded1`. If `main` drifts unexpectedly,
diff against that commit to recover the known-good baseline.

## Success criteria for "continuation succeeded"

A future chat has successfully continued this project if and only if:

1. `pytest -q` still reports 104 passing tests (or more, never fewer
   without an explicit deletion memo).
2. The smokes in the **Next 20 minutes** section run to completion on CPU
   and produce the same shape of JSONL output as today.
3. Every new claim about emergent communication, transfer, or
   open-endedness is paired with the corresponding ablation / shadow /
   transfer evidence already wired into the repo. No new unsupported
   adjectives in `README.md` or `docs/*.md`.
4. New sprint branches follow the `sprint/<topic>-fast` pattern, land via
   PR into `main`, and update `docs/CONTINUATION_HANDOFF.md` if they
   change project state that a future zero-context reader would need.
5. Headline numbers in any new memo are reported with seeds, horizon, and
   compute budget — never as point estimates.
