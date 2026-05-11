# Experiment Results Report Template

> Copy this template into your experiment's run directory (e.g.
> `outputs/runs/<RUN_NAME>/report.md`) and fill in the sections below.
> The companion script `scripts/generate_report.py` produces a first-pass
> auto-filled draft; this template is the human-curated, publication-ready
> superset.

The template is **continuation-safe**: every section has a `TODO` placeholder
so a partially filled report still renders cleanly, and downstream tooling
(diffs, PR previews, cross-run comparisons) can be appended without
restructuring the document.

---

## 0. Run metadata

| field | value |
|---|---|
| run name | `<RUN_NAME>` |
| git commit | `<SHA>` |
| config path | `<configs/...yaml>` |
| seeds | `<list>` |
| date | `<YYYY-MM-DD>` |
| operator | `<name / agent id>` |
| compute | `<device, wallclock>` |

---

## 1. Validation status

State the **pre-registered hypothesis** and the **decision rule** before
inspecting numbers. See `docs/preregistration_template.md` and
`docs/scientific_validation.md`.

- [ ] Hypothesis registered before run? (`docs/preregistration_template.md`)
- [ ] Seeds ≥ 10 for headline claims? (`scripts/run_ablation_matrix.py --seeds 10`)
- [ ] Baseline matched in compute and wallclock?
- [ ] Statistical test specified (median + IQR + Cliff's δ)?
- [ ] Language guardrails followed? (no "evolution"/"language"/"tool use" without §7 evidence)

**Validation verdict:** `pending` | `passes` | `fails` | `inconclusive`

TODO: one-paragraph plain-English summary of what was tested and what the
data say about the hypothesis.

---

## 2. Headline metrics

Auto-fillable from `metrics.jsonl` via `generate_report.py --headline`.

| metric | average | final | baseline avg | Δ | Cliff's δ |
|---|---|---|---|---|---|
| `embodied_alive_frac` | | | | | |
| `avida_alive_frac` | | | | | |
| `embodied_mean_lineage_depth` | | | | | |
| `embodied_lineage_hill1d` | | | | | |
| `avida_tasks_solved` | | | | | |
| `comm_usage_rate` | | | | | |
| `coordinated_behavior_index` | | | | | |

TODO: comment on each row that moved by more than the pre-registered effect
size threshold.

---

## 3. Scaling and transfer (if applicable)

Auto-fillable from `scaling_slopes.json` and `transfer_matrix.json`.

### 3.1 Compute-scaling slopes

| axis | slope | r² | CI95 |
|---|---|---|---|
| population | | | |
| world size | | | |
| generations | | | |

TODO: identify which axis returns the best metric per FLOP and call out any
plateaus.

### 3.2 POET-style transfer matrix

Source → target task transfer (higher = better, diagonal = self-play):

```
<paste transfer_matrix.json summary table here>
```

TODO: comment on off-diagonal asymmetries — they are the interesting signal.

---

## 4. QD / novelty archive (if applicable)

Auto-fillable from `map_elites.npz` and/or `novelty_archive.npz`.

- archive coverage: `<n_filled>/<n_cells>` (`<fraction>`)
- QD score: `<sum of cell fitness>`
- novelty archive size: `<n_entries>`
- mean k-NN novelty: `<value>`

TODO: include the `plots/map_elites.png` heatmap reference and call out any
empty regions of the behavior space.

---

## 5. Ablation comparison

For each ablation in the matrix (see `scripts/run_ablation_matrix.py`):

| condition | seeds | median headline | IQR | δ vs control | passes threshold? |
|---|---|---|---|---|---|
| control | | | | — | — |
| `<ablation A>` | | | | | |
| `<ablation B>` | | | | | |

TODO: one bullet per ablation: what was removed, what changed, whether the
direction matches the pre-registered prediction.

---

## 6. Scientific caveats

Cross-reference `docs/scientific_validation.md`. Default disclaimers — keep
these even if numbers look good:

- This report is **descriptive**. None of the numbers constitute evidence
  of *open-ended evolution*, *language*, or *tool use* — see
  `docs/scientific_validation.md` §7 for the language guardrails.
- Single-seed numbers are **pilots only**. Headline claims require
  `--seeds 10` (or whatever the pre-registration specified) and the
  median + IQR + Cliff's δ reporting recipe.
- The proxy field is an **inductive bias**, not a fluid simulator. Treat
  any "physics" wording accordingly.
- Comm-usage metrics are easy to inflate with reward shaping; check
  `enrichment_separation` and the `comm_benchmark.md` controls before
  claiming communication.
- Compute-scaling fits over fewer than ~5 points are **anecdotes**, not
  power laws. Report CI95 and r².

TODO: add caveats specific to this run (e.g. shortened wallclock, modified
mutation kernel, partial archive).

---

## 7. Next experiments

List concrete follow-ups, each with a (a) pre-registered hypothesis, (b)
seed/compute budget, (c) decision rule. Aim for ≤3.

1. TODO
2. TODO
3. TODO

---

## 8. Artifacts index

Paths are relative to the run directory.

- `metrics.jsonl` — per-step / per-generation metrics stream
- `scaling_slopes.json` — compute-scaling regression outputs (optional)
- `transfer_matrix.json` — POET-style transfer matrix (optional)
- `map_elites.npz` — MAP-Elites archive (optional)
- `novelty_archive.npz` — novelty archive (optional)
- `checkpoint_final.pkl` — final world + populations
- `plots/` — auto-generated PNGs (`world_fields`, `agents`, `metrics`,
  `map_elites`)
- `report.md` — this file
