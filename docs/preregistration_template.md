# Pre-Registration Template

Fill in before launching a release-quality experiment. Commit the populated
file to `experiments/<run-name>.md` *before* the first run completes — this
is what distinguishes pre-registered descriptors from cherry-picked ones
(memo §3 failure modes).

---

## 1. Research question

> One sentence. e.g. "Does adding curvature-driven enrichment increase the
> effective number of lineages above a neutral-shadow baseline?"

## 2. Hypothesis

> Direction + magnitude + statistical threshold. e.g. "Median lineage Hill-1D
> is ≥ 1.5× the shadow baseline at generation 50 with Cliff's δ ≥ 0.4."

## 3. Behavior descriptors (MAP-Elites / novelty)

| Descriptor | Source | Bounds | Pre-registered? |
|---|---|---|---|
| e.g. mean_x_position | EmbodiedPopulationState.behavior_descriptor[..., 0] | [0, 1] | yes |
| e.g. action_entropy | metrics.core.action_entropy | [0, log 8] | yes |
| e.g. <post-hoc descriptor> | — | — | **post-hoc — flag** |

## 4. Minimal criterion

State explicitly and *do not change mid-run*. Common form:
"An agent counts as surviving iff `energy >= 0` AND `age <= MAX_AGE` at
end-of-generation."

## 5. Conditions

| Name | Config | Selection? | Notes |
|---|---|---|---|
| baseline | configs/base.yaml | yes | reference |
| shadow | configs/base.yaml (selection patched off) | no | Bedau control |
| ablation-A | configs/ablation_no_comms.yaml | yes | comms channel zeroed |
| ablation-B | configs/ablation_static_world.yaml | yes | physics-as-decoration check |

## 6. Seeds

- Number of seeds per condition: **___** (≥ 10 for headline; ≥ 3 for pilot).
- Range / generation: e.g. `range(0, 10)`.

## 7. Headline metrics & decision rules

| Metric | Direction | Test | Pass threshold |
|---|---|---|---|
| `embodied_lineage_hill1d` | larger | Cliff's δ vs shadow | δ ≥ 0.4 |
| `qd_score` | larger | Cliff's δ vs baseline | δ ≥ 0.3 |
| `topsim` (if claiming communication) | larger | vs shuffled-channel | δ ≥ 0.3 *and* held-out generalisation > random |

## 8. Stopping rule

Wall-clock or generation budget; commit before running. e.g. "Stop at
generation 50 or 6 wall-hours, whichever first."

## 9. Reporting commitments

- [ ] Median + IQR + 95% CI for every metric in the headline table.
- [ ] Effect size on every comparison.
- [ ] Limitations + anti-overclaim section (`scripts/generate_report.py`
      enforces a placeholder).
- [ ] Filled `EC_REPRODUCIBILITY_CHECKLIST.md` copied to
      `outputs/<run>/`.

## 10. Deviations log

Record every deviation from this pre-registration after the fact — *do not
silently edit it*. Append-only.

| Date | Section | Change | Reason |
|---|---|---|---|
| | | | |
