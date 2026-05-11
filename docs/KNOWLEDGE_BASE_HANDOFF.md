# Knowledge-Base Handoff — hybrid-alife

> Single-page, knowledge-base-ready snapshot of project state at the close of the
> final sprint. Designed to be (a) dropped into a KB / RAG index as discrete
> bullet facts, and (b) the first file a new chat reads to continue work with
> zero prior context. Pair with [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md)
> for the longer narrative.

## Project identity

- **Name:** hybrid-alife — Hybrid Artificial-Life Simulator.
- **Repository:** https://github.com/sebastbernal2-ship-it/hybrid-alife
- **Main branch tip at handoff:** `b98123307cd4367f70bcef9ba593733cc1ad5a6b`
  (`merge: visual run display tooling`).
- **This sprint branch:** `sprint/kb-handoff-final` (docs-only).
- **Date of handoff:** 2026-05-11.
- **License / status:** research code, CPU-feasible smokes, GitHub Actions CI.

## Mission (one paragraph)

A JAX research simulator that couples **embodied recurrent neural agents** (GRU /
MLP controllers) with **Avida-style instruction-genome organisms** in a shared 2D
world built from microfluidics-inspired proxy fields — Dean-flow curvature,
shear, shear gradient, inertial-lift proxy, margination/enrichment, diffusing
concentration, and metabolites. The goal is not to simulate a wet lab; it is to
provide structured, partially-observable sensory pressure for open-ended
evolution, then measure that openness with Bedau activity, MAP-Elites coverage /
QD-score, compositionality (topsim / posdis / bosdis), Hill 1D effective lineage
count, and transfer summaries — all paired with ablations and a neutral shadow.

## Implemented modules

```
src/hybrid_alife/
├── agents/        controller.py (GRU+MLP), embodied.py, avida_vm.py
├── world/         env.py (terrain, proxy fields, advection), sensing.py
├── evolution/     selection.py (tournament+reseed), archives.py (novelty+ME)
├── experiments/   runner.py, shadow.py, transfer.py
├── metrics/       core, bedau, qd, communication, comm_benchmark, lineage
├── logging/       JSONL writer
├── replay/        deterministic pickle checkpoints
├── viz/           matplotlib plotting helpers
└── types.py       shared dataclasses / configs

scripts/          run_sim, generate_report, run_ablation_matrix,
                  run_comm_benchmark, run_compute_scaling,
                  run_transfer_matrix, run_quick_campaign,
                  preflight_campaign, visualize_run
configs/          base, smoke200, 11 ablations, qd_active, comm_task,
                  lineage_growth, scaling_tiny, transfer_{source,target},
                  campaigns/ (multi-config sweeps)
tests/            CPU-feasible pytest suite
```

## Validation status (current)

- `pytest -q` → **193 passed in 98.20s** on CPU (full suite, no skips).
- Communication benchmark: compositional control hits topsim / posdis / bosdis
  = 1.0.
- QD-active smoke (`configs/qd_active.yaml`, gen 5): coverage 0.4375,
  qd_score 77.9649, novelty archive size 72.
- Compute scaling (budget 6, gen 5): coverage 0.5, qd_score 15.4245, novelty 36.
- Headline numbers above come from gen-5 / smoke configs. Anything publishable
  needs ≥10 seeds and longer horizons (see `SCIENTIFIC_INTERPRETATION_GUIDE.md`).

## Documentation map (final)

| File | What it is for |
|---|---|
| [`../README.md`](../README.md) | Top-level overview, quick start, repo layout. |
| [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md) | Long-form handoff: state, risks, next actions, recovery. |
| [`KNOWLEDGE_BASE_HANDOFF.md`](KNOWLEDGE_BASE_HANDOFF.md) | **This file** — one-page KB snapshot. |
| [`architecture.md`](architecture.md) | System architecture, data flow. |
| [`world_model.md`](world_model.md) | World state, proxy fields, advection. |
| [`agent_branches.md`](agent_branches.md) | Embodied + Avida branches in detail. |
| [`scientific_validation.md`](scientific_validation.md) | Metric definitions, ablation logic, claim guardrails. |
| [`SCIENTIFIC_INTERPRETATION_GUIDE.md`](SCIENTIFIC_INTERPRETATION_GUIDE.md) | Evidence tiers, allowed vs disallowed claims, calibration. |
| [`qd_active.md`](qd_active.md) | MAP-Elites archive, QD logging. |
| [`comm_benchmark.md`](comm_benchmark.md) | Synthetic compositionality benchmark. |
| [`lineage_growth.md`](lineage_growth.md) | Lineage tree, Hill 1D effective count. |
| [`poet_transfer.md`](poet_transfer.md) | v1 transfer matrix + compute scaling. |
| [`campaign_cache_acceleration.md`](campaign_cache_acceleration.md) | Cache-accelerated campaigns. |
| [`experiment_campaign_quickstart.md`](experiment_campaign_quickstart.md) | Multi-config sweep quickstart. |
| [`visualization_quickstart.md`](visualization_quickstart.md) | `visualize_run.py` usage. |
| [`LONG_RUN_OPERATOR_RUNBOOK.md`](LONG_RUN_OPERATOR_RUNBOOK.md) | Preflight, tmux, resume, 20-min/2-hr/overnight recipes. |
| [`GALLERY.md`](GALLERY.md) | Example runs and artifacts. |
| [`RESULTS_REPORT_TEMPLATE.md`](RESULTS_REPORT_TEMPLATE.md) | Report scaffold for write-ups. |
| [`preregistration_template.md`](preregistration_template.md) | Preregistration template. |
| [`CI_VALIDATION.md`](CI_VALIDATION.md) | CI gates and validation policy. |
| [`../EC_REPRODUCIBILITY_CHECKLIST.md`](../EC_REPRODUCIBILITY_CHECKLIST.md) | EC-style reproducibility checklist. |

## Exact commands

```bash
# Install (CPU-feasible; JAX CPU wheel)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Tests (expect 193 passing in ~98 s on CPU)
pytest -q

# Preflight before any multi-hour campaign
python scripts/preflight_campaign.py --config configs/campaigns/<campaign>.yaml

# Cache-accelerated quick campaign
python scripts/run_quick_campaign.py --config configs/campaigns/<campaign>.yaml

# Full campaign / ablation matrix with stats and neutral shadow
python scripts/run_ablation_matrix.py \
    --configs configs/smoke200.yaml configs/ablation_no_comms.yaml \
              configs/ablation_static_world.yaml configs/ablation_uniform_field.yaml \
    --seeds 3 --include-shadow \
    --out-dir outputs/ablation_matrix

# Single runs
python scripts/run_sim.py --config configs/base.yaml          # 2 gens
python scripts/run_sim.py --config configs/smoke200.yaml      # 200 gens, ~110 s
python scripts/run_sim.py --config configs/qd_active.yaml
python scripts/run_sim.py --config configs/comm_task.yaml
python scripts/run_sim.py --config configs/lineage_growth.yaml

# Benchmarks
python scripts/run_comm_benchmark.py
python scripts/run_compute_scaling.py
python scripts/run_transfer_matrix.py \
    --source configs/transfer_source.yaml \
    --target configs/transfer_target_uniform.yaml

# Visualize a run (per-step frames / overview plots)
python scripts/visualize_run.py outputs/runs/<run_name>

# Build markdown report from a run directory
python scripts/generate_report.py outputs/runs/<run_name>
```

## Artifact names (shared in this chat)

The repository writes to `outputs/` (gitignored). Expected artifact names a new
session should look for:

- `outputs/runs/<run_name>/metrics.jsonl` — per-step / per-gen metrics stream.
- `outputs/runs/<run_name>/config.yaml` — frozen config copy.
- `outputs/runs/<run_name>/checkpoints/*.pkl` — deterministic pickle checkpoints.
- `outputs/runs/<run_name>/report.md` + `*.png` — `generate_report.py` output.
- `outputs/runs/<run_name>/lineage.json` — lineage tree export.
- `outputs/runs/<run_name>/archive_mapelites.json` — MAP-Elites archive snapshot.
- `outputs/ablation_matrix/...` — multi-config × multi-seed sweep results,
  including paired neutral-shadow runs when `--include-shadow` is used.
- `outputs/comm_benchmark/...` — synthetic compositionality benchmark results.
- `outputs/compute_scaling/...` — budget × coverage scaling curves.
- `outputs/transfer/...` — v1 transfer matrix tables + bootstrap CIs.
- `outputs/campaigns/<campaign>/...` — cache-accelerated campaign artifacts.
- `outputs/visualizations/<run_name>/...` — frames and overview plots from
  `visualize_run.py`.

## Next-session first prompt

Copy-paste this into a new chat to continue work with zero prior context:

> You are continuing the hybrid-alife project at
> https://github.com/sebastbernal2-ship-it/hybrid-alife (main at
> `b98123307cd4367f70bcef9ba593733cc1ad5a6b`). First read
> `docs/KNOWLEDGE_BASE_HANDOFF.md`, then `docs/CONTINUATION_HANDOFF.md`, then
> `docs/SCIENTIFIC_INTERPRETATION_GUIDE.md`. Confirm baseline with
> `pytest -q` (expect 193 passing in ~98 s on CPU). Then pick one of the
> "Next priorities" items from `KNOWLEDGE_BASE_HANDOFF.md`, open a new branch
> `sprint/<topic>-fast` off main, and land it via PR with CI green. Do not
> weaken language guardrails or remove ablation pairings. Report seeds,
> horizon, and compute budget with every headline number.

## Knowledge-base bullet facts

Each bullet below is a self-contained fact suitable for a KB / RAG chunk.

- hybrid-alife is a JAX-based artificial-life simulator with two coupled
  branches: embodied GRU/MLP agents and Avida-style instruction-genome organisms.
- The shared `WorldState` contains terrain, resources, hazards, flow, shear,
  shear gradient, inertial-lift proxy, margination/enrichment, curvature,
  concentration, metabolites, occupancy, and time.
- The "sixth sense" comprises six modalities sampled at each agent's grid cell
  with optional noise, blind, shuffle, delay, and dropout ablations.
- Evolution uses tournament selection with reproductive reseeding into dead
  slots, a novelty archive, and a MAP-Elites archive with per-generation QD
  logging.
- Scientific metrics implemented: Bedau-Packard activity (A_new, A_cum, A_p)
  with paired neutral shadow; QD-score, coverage, archive entropy;
  topsim / posdis / bosdis on argmax-discretised messages plus MI channel
  capacity; Hill 1D effective lineage count.
- v1 transfer matrix re-evaluates per-config end-of-run metrics; true
  trained-on-A-evaluated-on-B transfer is not yet implemented.
- Compositionality scores on continuous channels are upper-bounded by argmax
  quantisation; topsim/posdis/bosdis = 1.0 on the synthetic benchmark validates
  the tooling, not emergent language.
- Reseeding hides extinction; embodied alive-fraction stays at 1.0, so
  selection pressure must be measured via lineage depth and Hill 1D, not
  survival fraction.
- The full pytest suite currently reports 193 passed in 98.20 s on CPU.
- GitHub Actions CI runs the full pytest suite on every push; merges into
  `main` require a green CI.
- Sprint branches use the pattern `sprint/<topic>-fast`; they land via merge
  commits and the chronology is `git log --oneline main`.
- Long campaigns must follow `docs/LONG_RUN_OPERATOR_RUNBOOK.md` — preflight,
  tmux session, `--resume` policy, 20-min / 2-hour / overnight recipes.
- `scripts/preflight_campaign.py` validates configs and environment before a
  multi-hour campaign starts.
- `scripts/run_quick_campaign.py` provides cache-accelerated multi-config
  sweeps (see `docs/campaign_cache_acceleration.md`).
- `scripts/visualize_run.py` renders per-step frames and overview plots from
  any `outputs/runs/<run_name>` directory.
- The proxy microfluidic fields are stylized inductive bias, not physical
  ground truth; `ablation_uniform_field` exists to demonstrate they are doing
  measurable work.
- Bedau activity ratios are only meaningful when the neutral shadow runner is
  paired to the live config; ablation matrices should be launched with
  `--include-shadow`.
- Language guardrails: `docs/scientific_validation.md` §7 commits to talking
  about labels and dynamics, not biology or linguistics; future PRs must not
  weaken this without an explicit memo update.

## Open risks (carry forward)

1. Proxy fields are stylized, not physical — claims about microfluidic-style
   ecology require the `ablation_uniform_field` comparison.
2. Compositionality on continuous channels is upper-bounded by argmax
   quantisation; numbers are lower bounds, not language evidence.
3. v1 transfer is re-evaluation, not transfer learning; a generalisation claim
   needs trained-on-A-evaluated-on-B with identical controller weights.
4. CI smokes are tiny (gen-5); variance across seeds is large at this scale —
   publishable numbers need ≥10 seeds and longer horizons.
5. Bedau ratios depend on a shadow paired to the live config; always pass
   `--include-shadow` when running the ablation matrix.
6. Language guardrails (`scientific_validation.md` §7) must not be weakened
   silently.
7. Reseeding hides extinction; alive-fraction is not a selection signal.

## Next priorities (in order)

1. **v2 transfer:** load policies from run-A checkpoints and evaluate them in
   config B without further mutation, behind a `--mode {reeval,transfer}` flag
   on `scripts/run_transfer_matrix.py`. Acceptance: bootstrap-CI on transfer
   gap vs reeval gap, both reported.
2. **Surrogate-assisted QD:** small regression head over behaviour-descriptor
   → fitness to gate full rollouts. Acceptance: ≥30% reduction in rollouts to
   reach current smoke200 coverage at matched seeds.
3. **Continuous-channel compositionality:** non-discretising estimator (e.g.
   kernel topsim) reported alongside the argmax-discretised baseline.
4. **POET-loop scaffold:** environment-generator population with a minimal
   criterion and an environment-archive paralleling the agent MAP-Elites
   archive.
5. **Headline at scale:** rerun smoke200 and one ablation at ≥10 seeds and the
   longest horizon CI can absorb; report seeds + horizon + compute budget with
   every number.

## Success criteria for "continuation succeeded"

1. `pytest -q` still reports ≥193 passing tests, never fewer without an
   explicit deletion memo.
2. The smokes in [`CONTINUATION_HANDOFF.md`](CONTINUATION_HANDOFF.md) "Next 20
   minutes" run end-to-end on CPU and produce the same JSONL shape.
3. Every new claim about emergent communication, transfer, or open-endedness
   is paired with the corresponding ablation / shadow / transfer evidence
   already wired into the repo.
4. New sprint branches follow `sprint/<topic>-fast`, land via PR into `main`
   with green CI, and update this file plus `CONTINUATION_HANDOFF.md` if they
   change project state a zero-context reader would need.
5. Headline numbers in any new memo are reported with seeds, horizon, and
   compute budget — never as point estimates.
