# Scientific Interpretation Guide

Status: **WORK IN PROGRESS** — initial skeleton. See bottom of file for limitations.

This guide tells you how to read results produced by this repository without
overclaiming. It is the canonical reference for what a given metric value means,
which claims the evidence supports, and which claims it does not.

If a paragraph, README section, blog post, or PR description seems to make a
stronger claim than this guide permits, the guide wins. Tighten the claim.

---

## 1. Audience and Scope

This guide is for:

- contributors writing up results from campaign runs
- reviewers checking whether a write-up is calibrated
- downstream readers trying to understand what a number means

It is **not** a tutorial for running the system (see
`docs/experiment_campaign_quickstart.md`) and not a methodology spec (see
`docs/scientific_validation.md`, `docs/preregistration_template.md`).

---

## 2. Evidence Tiers

All claims in write-ups should be tagged with one of the following tiers. The
tier is determined by the *weakest* link in the chain (seeds, ablations,
replication, baseline comparison).

### Tier E — Engineering Validation
The code runs end to end, metrics are produced, smoke tests pass.

- ✅ Allowed: "the pipeline executes", "the metric is computed without error",
  "shapes/dtypes are correct", "the cache hit rate on rerun is ≥ X".
- ❌ Not allowed: any claim about *what the system learned* or *what behavior
  emerged*.

### Tier P — Preliminary Evidence
A small number of seeds (typically 1–3) show a directionally consistent effect
against at least one ablation or baseline.

- ✅ Allowed: "in N=3 seeds, condition A produced higher score than ablation
  B", "an effect is visible but not yet quantified".
- ❌ Not allowed: "X causes Y", "X outperforms Y", point estimates without
  uncertainty, comparisons to external systems.

### Tier S — Strong Evidence
≥ 10 seeds per condition, with at least one preregistered ablation, effect
size reported with confidence interval, and an independent replication
(rerun-from-scratch by someone other than the original author, or a separate
machine with the cache cleared).

- ✅ Allowed: causal language constrained to the manipulated variable, effect
  size reporting, comparisons within the system.
- ❌ Not allowed: claims of generality beyond the tested environments,
  comparisons to systems we did not run ourselves, claims about
  human-relevant cognition / consciousness / agency.

A finding cannot be promoted from P to S by rerunning more seeds on the same
machine with the same cache — see §6.

---

## 3. Per-Metric Interpretation

Below is a per-metric crib sheet. Each entry lists: what the metric measures,
what increases/decreases mean, the floor below which the metric is noise, the
tier of claim it can support on its own, and the most common
misinterpretation.

> The numeric thresholds below are **placeholders** based on current campaign
> defaults. Replace them with calibrated values once a baseline distribution
> is available — see §9.

### 3.1 Lineage growth
- Measures: descendant count and depth of surviving lineages over a campaign.
- Increase: more stable reproduction within the run, *not* "intelligence".
- Noise floor: differences < ~10% across seeds within the same condition
  should be treated as noise.
- Supports: tier P on its own; tier S only if paired with an ablation that
  disables the mechanism being credited.
- Common misread: "lineages grew, so the agents learned X". Lineage growth
  can be driven entirely by environment forgiveness or by a degenerate
  strategy.

### 3.2 QD coverage / archive fill
- Measures: fraction of behavior-descriptor cells filled in the MAP-Elites
  style archive.
- Increase: more behavioral diversity *as defined by the descriptor*. This is
  a property of the descriptor as much as of the agents.
- Noise floor: archive fill within ±1 cell is noise; use the active-cells
  count from `qd_active`.
- Supports: tier P. Promoting to S requires showing that the descriptor is
  not trivially satisfiable (an "all-zero behavior" ablation should fail to
  fill cells).
- Common misread: "the agents found diverse strategies". They found diverse
  *descriptor values*, which may or may not correspond to strategies.

### 3.3 Communication benchmark score
- Measures: task success rate on the comm benchmark suite.
- Increase: agents pass more comm tasks. Says nothing about whether the
  protocol generalizes.
- Noise floor: a single seed swing of ≥ 1 task is within run-to-run variance
  on small suites.
- Supports: tier P with N ≥ 3 seeds. Tier S requires a held-out task split
  not used during selection.
- Common misread: treating the score as evidence of "language" or
  "compositionality". The benchmark tests task completion, not linguistic
  structure.

### 3.4 POET transfer score
- Measures: performance of an agent evolved in environment A when evaluated
  in environment B.
- Increase: better cross-environment transfer.
- Noise floor: gains < the within-environment seed variance are noise. Always
  report within-env variance alongside transfer numbers.
- Supports: tier P. Tier S requires both directions (A→B and B→A) and
  multiple environment pairs.
- Common misread: "the agents generalize". Transfer between two
  procedurally-related environments is not generalization.

### 3.5 World-model prediction error
- Measures: rollout prediction error of the learned world model.
- Decrease: better one-step (or k-step) predictions.
- Noise floor: differences within the per-batch standard error are noise.
- Supports: tier P. Lower error does not imply better downstream control;
  pair with a control-task evaluation for tier S.
- Common misread: "the world model understands the environment". It
  minimizes a loss.

### 3.6 Cache hit rate (campaign cache)
- Measures: fraction of campaign artifacts retrieved from cache rather than
  recomputed.
- This is an **engineering** metric (tier E). It tells you nothing about the
  science. A high cache hit rate on a rerun is expected and good for
  reproducibility cost, but it does **not** count as replication (see §6).

---

## 4. Allowed vs. Disallowed Claims

### Allowed
- Comparisons *within a single campaign* between a condition and its
  preregistered ablation, with seeds reported.
- Statements about which mechanism, *when removed*, degrades which metric.
- Statements about pipeline correctness, runtime, and reproducibility.
- Statements about behavioral diversity *under a specific descriptor*.

### Disallowed without explicit Tier-S backing
- "Emergent" anything. The word "emergent" requires (a) a baseline where the
  property is absent, (b) a manipulation that turns it on, (c) ≥ 10 seeds.
- "Open-ended". Open-endedness is a long-horizon property; a single campaign
  cannot demonstrate it.
- "Intelligent", "understands", "learns to reason". Use behavioral
  descriptions instead ("achieves task X under condition Y").
- Comparisons to published numbers from other systems unless we reran those
  systems under matched conditions.
- Claims about cognition, consciousness, agency, intent.
- Extrapolation beyond the tested environments, scales, or budgets.

### Always required when stating a result
- N (seeds), the ablation it is compared against, the metric, the effect
  direction, and an uncertainty estimate (CI, IQR, or per-seed values).
- The git SHA of the code that produced the numbers.
- A pointer to the raw artifacts (campaign id / cache key / run directory).

---

## 5. Ablation Logic

An ablation is a *minimal* change that removes the mechanism being credited.
It must be:

1. **Preregistered** in `docs/preregistration_template.md` before the run.
2. **Minimal** — change one mechanism, hold everything else fixed (seed
   schedule, compute budget, env distribution, eval protocol).
3. **Symmetric in budget** — the ablation must get the same wall-clock or
   step budget as the full condition. Starving the ablation is a common
   silent bias.
4. **Result-blind in design** — the ablation cannot be chosen after seeing
   the result it would produce.

A claim of the form "mechanism M contributes to metric X" requires:

- the full condition with M,
- the ablation without M (or with M replaced by a trivial substitute),
- ≥ 3 seeds for tier P, ≥ 10 seeds and reported effect size for tier S,
- a statement of which metric was *pre-declared* the primary outcome (to
  avoid multiple-comparisons fishing).

If multiple metrics are reported, only the preregistered primary is eligible
for a causal claim; the rest are exploratory and must be labeled as such.

---

## 6. Replication Requirements

Replication is **not** the same as "rerunning the same script on the same
machine". By tier:

- **Tier E**: rerun on the same machine, cache cleared, must produce
  bit-identical or numerically-equivalent artifacts. Verifies the pipeline,
  not the science.
- **Tier P**: rerun by a different person OR on a different machine, with
  the cache cleared and the seed schedule preserved. Effect direction must
  reproduce.
- **Tier S**: independent replication — different person, different machine,
  cache cleared, *and* a second seed schedule drawn from the same
  distribution. Effect direction and rough magnitude must reproduce.

Cache hits count toward engineering reproducibility (tier E) only. A campaign
with 100% cache hits is a zero-evidence replication.

---

## 7. Failure Modes to Watch For

These are the most common ways a result becomes overclaimed in practice in
this repo. Reviewers should actively look for each one.

1. **Seed cherry-picking.** Reporting the best of N seeds without disclosing
   N or the distribution. Fix: report all seeds or an aggregate with
   variance.
2. **Ablation budget asymmetry.** The full condition gets more compute, more
   steps, or more eval episodes than the ablation. Fix: enforce identical
   budgets via the campaign config.
3. **Descriptor leakage in QD.** The behavior descriptor encodes the
   evaluation signal, so archive fill correlates with reward trivially.
   Fix: descriptor must be definable without reading the reward.
4. **Eval-on-train.** Transfer or generalization scores computed on
   environments that were in the training distribution. Fix: explicitly
   declare and hold out an eval split.
5. **Metric drift.** The metric definition changes between conditions
   (e.g., different episode length, different scoring rule). Fix: pin
   eval config separately from training config.
6. **Multiple comparisons without correction.** Reporting the best of K
   metrics as "the" effect. Fix: preregister a single primary metric;
   everything else is exploratory.
7. **Cache-confounded comparison.** Conditions A and B differ in cache
   hit rate, so wall-clock comparisons are misleading. Fix: report
   compute in normalized units (env steps, model FLOPs), not wall-clock.
8. **Anthropomorphic framing.** Using words like "learns", "decides",
   "understands" in places where a behavioral description would do. Fix:
   rewrite in behavioral terms.
9. **Over-aggregation.** Averaging across heterogeneous environments
   hides the fact that one environment carries the entire effect. Fix:
   report per-environment numbers in addition to any aggregate.
10. **Cherry-picked checkpoint.** Reporting the best-ever value during a
    run rather than the final or a pre-declared checkpoint. Fix:
    pre-declare which checkpoint(s) count.

---

## 8. Concrete Thresholds (current defaults)

These are starting thresholds. They are *not* tuned to a measured baseline
distribution. Update them — and this section — once a baseline campaign with
≥ 10 seeds on the null/ablated condition exists. Until then, treat them as
upper bounds on credulity, not lower bounds on truth.

| Metric                       | Engineering (Tier E) | Preliminary (Tier P) | Strong (Tier S) |
| ---------------------------- | -------------------- | -------------------- | --------------- |
| Lineage growth (final depth) | runs without error   | ≥ 1.25× ablation, N ≥ 3, same direction in all seeds | ≥ 1.5× ablation, N ≥ 10, 95% CI excludes 1.0×, replicated |
| QD active cells              | metric is computed   | ≥ 1.2× ablation, N ≥ 3 | ≥ 1.5× ablation, N ≥ 10, descriptor non-trivial, replicated |
| Comm benchmark pass rate     | suite executes       | absolute gain ≥ 10 pp over ablation, N ≥ 3 | absolute gain ≥ 20 pp, N ≥ 10, held-out task split, replicated |
| POET transfer score          | both envs execute    | within-env variance < transfer gap, N ≥ 3, both directions reported | bidirectional, ≥ 2 env pairs, N ≥ 10, replicated |
| World-model prediction error | error is finite      | ≥ 15% relative reduction vs ablation, N ≥ 3 | ≥ 25% relative reduction, paired control-task gain, N ≥ 10, replicated |
| Cache hit rate (rerun)       | ≥ 0.9 on rerun       | (not applicable — engineering metric) | (not applicable) |

Numbers in this table are placeholders chosen to be deliberately
conservative. Calibrate them against a measured null distribution before
quoting them externally.

---

## 9. How to Calibrate the Thresholds

1. Pick the metric.
2. Run the *ablated* (mechanism-off) condition for ≥ 10 seeds.
3. Compute the seed-level distribution (mean, IQR, 95% range).
4. Tier P threshold = outside the IQR of the null.
5. Tier S threshold = outside the 95% range of the null, with N ≥ 10 in the
   full condition and a replication.
6. Record the calibration run's SHA, seed schedule, and artifact pointer in
   this file when you update the table.

Until step 2 has been done for a metric, that metric's thresholds in §8 are
provisional and any Tier-S claim using it is unsupported.

---

## 10. Cross-References

- `docs/scientific_validation.md` — broader validation methodology
- `docs/preregistration_template.md` — preregistering experiments
- `docs/experiment_campaign_quickstart.md` — running campaigns
- `docs/campaign_cache_acceleration.md` — cache semantics (relevant to §6)
- `docs/RESULTS_REPORT_TEMPLATE.md` — write-up template; should cite tiers
  defined here
- `docs/CONTINUATION_HANDOFF.md` — handoff notes for the next contributor

---

## 11. Limitations of This Guide

- Thresholds in §8 are placeholders, not calibrated against a measured null.
- The metric list in §3 covers the metrics surfaced by current campaigns; it
  is not exhaustive and will need extension as new metrics land.
- The guide assumes campaigns are run via the cached pipeline; ad-hoc runs
  outside the campaign harness may not produce the artifacts referenced
  here.
- "Replication" as defined in §6 requires a second person or machine. Solo
  contributors cannot, by definition, produce Tier-S evidence on their own
  and should not try to claim it.
- This document does not cover statistical tests (which test to use, how to
  correct for multiple comparisons) — only the structural requirements for a
  claim. Pair with a statistician or a stats reference before quoting
  p-values.
