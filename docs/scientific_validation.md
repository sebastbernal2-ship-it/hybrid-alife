# Hybrid-ALife: Scientific Validation & Technical Acceptance Memo

**Purpose.** Define what "works" means for the hybrid-alife repository — concrete metrics, experiments, failure modes, and claim-discipline guardrails — grounded in the canonical literature on embodied artificial life, digital evolution, quality-diversity search, emergent communication, and physics-inspired proxy fields.

**Scope.** Five technical pillars are treated as a single integrated validation surface: (1) JaxLife-style embodied agents, (2) Avida-lineage digital evolution, (3) novelty search / MAP-Elites / open-endedness, (4) emergent communication, (5) inertial-microfluidics-inspired proxy fields. Each pillar gets: required metrics, acceptance thresholds, expected failure modes, and language hygiene.

---

## 1. Pillar 1 — Embodied ALife (JaxLife-inspired)

### What the reference establishes
JaxLife is an end-to-end-differentiable, GPU-accelerated artificial life simulator in which embodied agents (parameterized by deep neural networks with attention + LSTM components) evolve via natural selection in a Turing-complete world, and were reported to develop "rudimentary communication protocols, agriculture, and tool use" with complexity scaling against compute ([Lu et al., arXiv:2409.00853](https://arxiv.org/abs/2409.00853)). Implementation details (per-entity encoders, self-attention over neighbors, terraforming actions, programmable robots, mutation-based inheritance) are documented in the [JaxLife GitHub repo](https://github.com/luchris429/jaxlife).

### Required metrics (must implement)
| Metric | Definition | Acceptance signal |
|---|---|---|
| **Population survival curve** | Median lifetime per generation | Monotone-rising or stable above neutral baseline |
| **Compute-vs-complexity curve** | A behavioral-complexity proxy (e.g. count of distinct action n-grams, attended-entity entropy) plotted vs FLOPs/steps | Positive slope at \( p<0.05 \) across ≥3 seeds, mirroring JaxLife's scaling claim ([Lu et al.](https://arxiv.org/html/2409.00853v1)) |
| **Generation depth** | Mean ancestral chain length of surviving lineages | Should exceed a non-selective control |
| **Environmental modification index** | Fraction of cells whose state was altered by agents (terraforming/agriculture proxy) | Required if claiming "niche construction" |

### Failure modes to test against
- **Reward hacking via boundary conditions** — agents discover toroidal corners or spawn-edge artifacts; check by ablating world topology.
- **Compute-scaling artifact** — complexity rises only because the *world* gets larger or noisier with compute. Hold world size fixed across scaling sweeps.
- **Lineage collapse** — apparent diversity is a single mutational hub. Track effective number of lineages (Hill number \( {}^1D \)) not raw species counts.

### Claim hygiene
Do **not** call behaviors "agriculture," "tool use," or "language" without an operational definition tied to the metric above. JaxLife itself uses the hedge "rudimentary" ([Lu et al.](https://arxiv.org/abs/2409.00853)) — match that register.

---

## 2. Pillar 2 — Digital Evolution Best Practices (Avida lineage)

### What the reference establishes
Avida is the canonical digital-evolution platform: self-replicating computer programs subject to mutation, competition, and selection, used for hypothesis-driven evolutionary experiments rather than as a toy ([Avida — ALife encyclopedia](https://alife.org/encyclopedia/digital-evolution/avida/); [Ofria et al., *Evolution Experiments with Self-Replicating Computer Programs*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7123229/)). The landmark Lenski-Ofria-Pennock-Adami study showed complex features (EQU logic) only evolve when simpler stepping-stone functions are also selectively favored — a result directly relevant to curriculum/reward design ([Lenski et al., *Nature* 2003](https://www.nature.com/articles/nature01568)).

### Required experimental discipline
1. **Pre-register a neutral (no-selection) control** alongside every selective run, per Bedau's activity-statistics methodology ([Bullock & Bedau, *Artificial Life* 2006](https://people.reed.edu/~mab/publications/papers/bullock.bedau.ALJ06.pdf)). Without this, any "novelty" claim is unfalsifiable.
2. **Stepping-stone analysis** — for each "complex" trait claimed, show the sub-trait ladder is *individually rewarded*; absent rewards collapse the trait, replicating the Lenski et al. finding ([Lenski et al., 2003](https://www.nature.com/articles/nature01568)).
3. **Reproducibility checklist** — adopt the 5-dimension EC reproducibility checklist (methodological clarity, experimental setup, results reporting, artifact evaluation, paper metadata) of [López-Ibáñez et al., arXiv:2602.07059](https://arxiv.org/html/2602.07059v1). At minimum: ≥10 seeds per condition, fixed RNG keys, environment versioning, full hyperparameter dump.

### Failure modes
- **Reporting only the best run.** Use median + IQR across seeds; report distribution shape.
- **Logic-task overfit.** Lenski-style EQU-style tasks are a *hypothesis testbed*, not a benchmark to beat; treat any score on a fixed task suite as descriptive, not as "progress."
- **Ontology drift.** When extending Avida-style organisms, follow [OntoAvida](https://icbo-conference.github.io/icbo2022/papers/ICBO-2022_paper_3396.pdf) naming so claims are comparable across papers.

---

## 3. Pillar 3 — Novelty Search, MAP-Elites, Open-Endedness

### What the references establish
- **Novelty search** abandons the objective function and rewards behavioral novelty alone, often outperforming objective-driven search on deceptive problems ([Lehman & Stanley, *Evolutionary Computation* 2011 / dissertation](https://joellehman.com/lehman-dissertation.pdf); [PubMed 20868264](https://pubmed.ncbi.nlm.nih.gov/20868264/)).
- **MAP-Elites** tiles a user-chosen *behavior descriptor space* into cells and keeps the elite per cell, producing a *map* (illumination) rather than a single best — yielding both diverse and high-performing solutions ([Mouret & Clune, arXiv:1504.04909](https://arxiv.org/abs/1504.04909); [Mouret QD page](https://members.loria.fr/jbmouret/qd.html); reference implementation pattern in [EvoTorch docs](https://docs.evotorch.ai/latest/examples/notebooks/Feature_Space_Illumination_with_MAPElites/)).
- **Open-endedness** is measured via Bedau-Packard *evolutionary activity statistics* (new components, persistence, mean cumulative activity) with a neutral-shadow baseline ([Bullock & Bedau 2006](https://people.reed.edu/~mab/publications/papers/bullock.bedau.ALJ06.pdf); [O'Reilly Radar overview](https://www.oreilly.com/radar/open-endedness-the-last-grand-challenge-youve-never-heard-of/); [Aston / Stepney review](https://research.aston.ac.uk/en/publications/insights-from-artificial-life-measuring-and-classifying-open-ende)).
- **POET** demonstrates the stepping-stone phenomenon empirically: coevolving environments + agents with periodic transfer beats direct optimization and direct curricula ([Wang et al., arXiv:1901.01753](https://arxiv.org/abs/1901.01753); [POET in ACM GECCO](https://dl.acm.org/doi/10.1145/3321707.3321799)).
- **Caveat literature**: Hickinbotham & Stepney argue any fixed open-endedness metric is eventually escaped by a truly open-ended system, so metrics describe rather than certify open-endedness ([Hickinbotham et al., *Artificial Life* 2024](https://direct.mit.edu/artl/article/30/3/390/114972/On-the-Open-Endedness-of-Detecting-Open-Endedness)). [Soros & Stanley](https://www.uvm.edu/neurobotics/pubs/pdf/2016_SorosCheneyStanley_HowTheStrictnessOfTheMinimalCriterionImpactsOpenEndedEvolution_ALIFE.pdf) further show the *strictness of the minimal criterion* heavily shapes open-endedness outcomes.

### Required metrics (must implement)
| Metric | Formal definition | Operationalization |
|---|---|---|
| **QD-score** | \( \sum_{c \in \text{filled cells}} f(c) \) over the MAP-Elites archive | Track per-generation; primary headline number for QD experiments |
| **Coverage** | filled-cell count / total cells | Distinguish "good" from "broad" performance |
| **Archive entropy** | Shannon entropy over occupied cells | Detects mode collapse in novelty/QD |
| **Behavioral-distance novelty score** | Mean k-NN distance in descriptor space (k=15 is standard from [Lehman & Stanley](https://joellehman.com/lehman-dissertation.pdf)) | Use as fitness in novelty-search ablation |
| **Bedau activity statistics** | New-component count \(A_{\text{new}}\), cumulative activity \(A_{\text{cum}}\), persistent-component count \(A_p\) | Compute on *both* the experimental run and a no-selection shadow ([Bullock & Bedau](https://people.reed.edu/~mab/publications/papers/bullock.bedau.ALJ06.pdf)) |
| **Stepping-stone transferability (POET-style)** | Probability that an agent evolved in env \(E_i\) solves env \(E_j\) after transfer | Required if claiming "open-ended coevolution" ([Wang et al.](https://arxiv.org/abs/1901.01753)) |

### Experimental design
- Run **three controls per QD experiment**: (a) pure objective search, (b) random search, (c) novelty-only search. Headline QD wins must beat all three on QD-score *and* not lose by >5% on best-fitness.
- Use ≥3 distinct behavior descriptors and report sensitivity — MAP-Elites results are descriptor-dependent ([Mouret & Clune](https://arxiv.org/abs/1504.04909)).

### Failure modes
- **Cherry-picked descriptor space** that *defines* novelty into existence. Mitigate by running with two pre-registered descriptors plus one chosen after seeing data; flag the last.
- **Open-endedness theater** — declaring open-endedness from a rising activity curve over a short window. Per [Hickinbotham et al.](https://direct.mit.edu/artl/article/30/3/390/114972/On-the-Open-Endedness-of-Detecting-Open-Endedness), only report *unbounded-trend evidence* with caveats; never claim "open-ended" as a binary property.
- **Minimal-criterion drift** — relaxing the survival criterion mid-run to "rescue" diversity inflates novelty. Lock the criterion and report only locked runs ([Soros & Stanley](https://www.uvm.edu/neurobotics/pubs/pdf/2016_SorosCheneyStanley_HowTheStrictnessOfTheMinimalCriterionImpactsOpenEndedEvolution_ALIFE.pdf)).

---

## 4. Pillar 4 — Emergent Communication Metrics

### What the references establish
The dominant quantitative proxy for compositionality in emergent communication is **topographic similarity (topsim)** — the Spearman/Pearson correlation between pairwise meaning distance and pairwise message distance ([Brighton & Kirby, *Artificial Life* 2006](https://pubmed.ncbi.nlm.nih.gov/16539767/); [Brighton & Kirby PDF](https://langev.com/pdf/brighton_visualizing_ALife.pdf)). However, [Chaabouni et al., arXiv:2004.09124](https://arxiv.org/pdf/2004.09124) show topsim is *insufficient*: emergent languages can generalize perfectly without high topsim, and topsim is agnostic to compositional *type*. They introduce **positional disentanglement (posdis)** and **bag-of-symbols disentanglement (bosdis)** as finer-grained measures. The Lazaridou et al. line shows that *input structure* drives protocol structure ([Lazaridou et al., arXiv:1804.03984](https://arxiv.org/abs/1804.03984)), and recent work links **limited data exposure / Zipfian frequency** to compositional pressure ([EMNLP 2025: Frequency & Compositionality](https://aclanthology.org/2025.emnlp-main.1387.pdf); [EMNLP 2024: One-to-Many Communication](https://aclanthology.org/2024.emnlp-main.1157.pdf)). For broader scope see [Peters & Martins, arXiv:2407.03302](https://arxiv.org/abs/2407.03302).

### Required metrics (must implement)
| Metric | Definition | Why include |
|---|---|---|
| **Topsim** | Spearman ρ between meaning- and message-distance matrices ([Brighton & Kirby](https://pubmed.ncbi.nlm.nih.gov/16539767/)) | Field-standard baseline; required for comparability |
| **Posdis** | Per-position symbol–attribute mutual-information gap ([Chaabouni et al.](https://arxiv.org/pdf/2004.09124)) | Detects order-based compositionality |
| **Bosdis** | Symbol-count–attribute MI gap ([Chaabouni et al.](https://arxiv.org/pdf/2004.09124)) | Detects order-free compositionality |
| **Generalization to held-out meaning combos** | Accuracy on novel attribute combinations | Required before any "compositional" claim |
| **Channel-capacity / mutual information** | \( I(\text{message}; \text{referent}) \) | Distinguishes communication from coincidence vs. positional cues |
| **Ablation: shuffled-channel control** | Replace message with random tokens; agents should fail | Demonstrates signal *is* being used |

### Failure modes
- **Cue leakage**: agents condition on shared observation timing or action history, not on messages. Always run a channel-shuffle ablation.
- **Topsim-only reporting**: per [Chaabouni et al.](https://arxiv.org/pdf/2004.09124) this can be high in non-compositional languages and low in genuinely generalizing ones. Report posdis + bosdis + held-out accuracy together.
- **Population-of-two artifact**: bilateral protocols overfit. Validate across ≥3 agent pairings and randomized partner-swap rounds.

### Claim hygiene
Avoid the words **"language"** and **"protocol with grammar"** unless held-out generalization > random *and* at least one of posdis/bosdis is significantly above a non-communicative baseline. Use "signaling system" or "discrete communication channel" otherwise.

---

## 5. Pillar 5 — Inertial Microfluidics as Proxy-Field Inspiration

### What the references establish
Inertial microfluidics is a regime where *fluid inertia*, normally negligible at microscale, produces robust **particle focusing** at finite particle Reynolds number \( Re_p = Re \cdot (a/D_h)^2 \), with secondary **Dean flows** in curved/spiral channels producing predictable equilibrium positions ([Di Carlo, *Lab on a Chip* 2009](https://pubs.rsc.org/en/content/articlelanding/2009/lc/b912547g); [Di Carlo, PubMed](https://pubmed.ncbi.nlm.nih.gov/19823716/); [Martel & Toner, *Annu. Rev. Biomed. Eng.* / PMC4467210](https://pmc.ncbi.nlm.nih.gov/articles/PMC4467210/); [Dean-flow dynamics, *Sci. Rep.* 2017](https://www.nature.com/articles/srep44072)). Critical-Stokes-number scaling further constrains particle-deposition probability ([Phillips & Sear, *PRF* 2023](https://link.aps.org/doi/10.1103/PhysRevFluids.8.014302)).

### Why this matters for hybrid-ALife
The relevance is not the application (cell sorting) but the **inductive bias**: a *low-dimensional, physically-grounded scalar/vector field* that produces emergent spatial ordering from local rules. As a *proxy field* in an ALife substrate this provides:
1. A non-uniform fitness/affordance landscape with closed-form structure (good for unit tests and ground-truth equilibria).
2. Multi-stable equilibrium positions — natural niches.
3. A continuous parameter (Re, Dean number \(De = Re \cdot \sqrt{D_h/2R}\)) that smoothly varies environmental "difficulty."

This connects to the broader pattern of **physics-informed neuroevolution**, where physical laws act as regularizers or curriculum scaffolds ([Sundar et al., arXiv:2501.06572](https://arxiv.org/html/2501.06572v2)).

### Required validation when using a proxy field
| Check | Acceptance |
|---|---|
| **Field correctness** | Equilibrium positions for test particles match published Di Carlo / Martel-Toner equilibria within tolerance ([Di Carlo](https://pubs.rsc.org/en/content/articlelanding/2009/lc/b912547g); [Martel & Toner](https://pmc.ncbi.nlm.nih.gov/articles/PMC4467210/)) |
| **Dimensional sanity** | Behavior changes correctly with Re, \(De\), \(a/D_h\) — failure here means the "physics" is decoration |
| **Critical-Stokes scaling** | If particle-like agents are used, deposition probability vs St shows the published \( \exp(-1/(St-St_c)^{1/2}) \) trend ([Phillips & Sear](https://link.aps.org/doi/10.1103/PhysRevFluids.8.014302)) |
| **Counterfactual ablation** | Replace the field with a uniform / random field; the emergent behaviors of interest should *change measurably*. If they do not, the physics is not doing work and should not be claimed. |

### Failure modes
- **Physics-as-aesthetic**: code uses microfluidics vocabulary but the field is just Perlin noise. Mitigate with the unit tests above.
- **Scale mismatch**: agent timestep and physical timestep are not reconciled, producing nonphysical equilibria. Document non-dimensionalization explicitly.
- **Overclaiming "physical realism"**: this is a *proxy field for inductive bias*, not a fluid simulator. State this in the README.

---

## 6. Cross-Cutting Acceptance Criteria for the Repo

### 6.1 Seeds, statistics, and reporting
- ≥10 seeds per condition for headline plots; ≥3 for pilot/ablation.
- Report **median and IQR**, not mean ± std, for any skewed or bounded metric (lifetimes, QD-score early in training).
- All headline differences require an effect size (Cliff's δ or Cohen's d) *and* a CI; bare p-values are insufficient. Apply Bonferroni or report family-wise error when ≥5 metrics are compared.
- Follow the [EC Reproducibility Checklist](https://arxiv.org/html/2602.07059v1) and ship its filled YAML alongside each release.

### 6.2 Logging schema (minimum)
Per generation, per seed:
- Population size, mean/median fitness, lineage tree id, mutation rate.
- QD-archive snapshot (cell, descriptor, fitness, genotype hash).
- Bedau activity vectors for both experimental and shadow runs.
- For communication runs: full meaning→message rollouts every K generations to recompute topsim/posdis/bosdis offline.
- Environment version hash + RNG seed + commit SHA.

### 6.3 Experiments to run before any external claim
1. **Baseline-vs-novelty-vs-QD sweep** on a fixed task suite (5 tasks × 10 seeds × 3 algorithms).
2. **Compute-scaling curve** holding world size fixed, varying training steps over ≥1.5 orders of magnitude (JaxLife pattern, [Lu et al.](https://arxiv.org/html/2409.00853v1)).
3. **Communication ablation matrix**: {channel on, channel shuffled, channel zeroed} × {symmetric, asymmetric tasks}.
4. **Proxy-field ablation**: {physics on, physics flat, physics random} × {agents fixed, agents evolved}.
5. **Open-endedness shadow study**: 1 selective + 1 neutral run per seed, ≥10 seeds, Bedau statistics reported jointly.

### 6.4 Failure-mode catalogue (consolidated)
| Mode | Detected by |
|---|---|
| Reward hacking | Heldout-environment evaluation; topology ablation |
| Lineage collapse | Effective lineage count (Hill \({}^1D\)) |
| Descriptor gaming | Pre-registered + post-hoc descriptors compared |
| Channel cue leakage | Shuffle/zero channel ablation |
| Physics-as-decoration | Uniform-field counterfactual |
| Cherry-picked seeds | Pre-registered seed range; full seed table in appendix |
| Open-endedness theater | Bedau shadow control; explicit caveat per [Hickinbotham et al.](https://direct.mit.edu/artl/article/30/3/390/114972/On-the-Open-Endedness-of-Detecting-Open-Endedness) |
| Topsim-only language claim | Posdis + bosdis + heldout-generalization required |

---

## 7. Language Guardrails (Anti-Overclaim Checklist)

Before any release note, paper, or blog post, every claim in the following table must pass:

| Tempting phrasing | Required evidence | If absent, say instead |
|---|---|---|
| "Open-ended evolution" | Bedau activity > shadow baseline, sustained over ≥N generations, with caveat | "Sustained novelty over the observed window" |
| "Emergent language" | Topsim **and** posdis or bosdis above non-communicative baseline + heldout accuracy | "Emergent signaling" or "discrete communication channel" |
| "Tool use" | Operational definition + counterfactual (remove tool ⇒ fitness drops) | "Object manipulation behavior" |
| "Agriculture" | Niche-construction index + heritable propagation | "Persistent environmental modification" |
| "Self-organization from physics" | Proxy-field ablation shows measurable behavior change | "Field-modulated behavior" |
| "Scales with compute" | Slope significant across seeds, world size held fixed | "Improves with training within tested regime" |
| "Outperforms" | ≥10 seeds, effect size + CI, multiple-comparison correction | "Higher median on this benchmark under these settings" |

[Hickinbotham et al.](https://direct.mit.edu/artl/article/30/3/390/114972/On-the-Open-Endedness-of-Detecting-Open-Endedness) note that open-endedness in particular is best treated as a *mechanism to study*, not a property to certify — adopt the same framing for every claim above.

---

## 8. Prioritized Recommendations for the Codebase

**P0 (block release until done):**
1. Implement Bedau activity statistics (\(A_{\text{new}}\), \(A_{\text{cum}}\), \(A_p\)) with a *paired neutral-shadow* run mode.
2. Implement MAP-Elites archive + QD-score + coverage + archive entropy, with at least two pluggable behavior descriptors.
3. Implement topsim, posdis, bosdis, and channel-shuffle ablation for any communication experiment.
4. Implement the four-cell proxy-field unit test (equilibrium positions, Re/De sensitivity, St scaling, uniform-field counterfactual).
5. Ship a filled [EC reproducibility checklist](https://arxiv.org/html/2602.07059v1) in the repo root.

**P1 (next milestone):**
6. POET-style transferability metric for any coevolutionary claim.
7. Effective lineage count (Hill \({}^1D\)) and lineage-tree export.
8. Pre-registration template for behavior descriptors and minimal criterion, committed before the run.
9. Compute-scaling harness that varies only training steps with world size locked.

**P2 (research extensions):**
10. Surrogate-assisted QD (per [Sundar et al.](https://arxiv.org/html/2501.06572v2)) to reduce cost of behavior-descriptor sweeps.
11. Meta-referential evaluation suite for compositionality stress tests ([Meta-Referential Games / S2B benchmark](https://openreview.net/forum?id=17BA0Tl2Id)).
12. OntoAvida-compatible naming for any digital-organism subsystem ([ICBO 2022 paper](https://icbo-conference.github.io/icbo2022/papers/ICBO-2022_paper_3396.pdf)).

---

## 9. One-Paragraph Executive Summary

The hybrid-alife repo should treat each of its five pillars — embodied agents, digital evolution, quality-diversity search, emergent communication, and physics-proxy fields — as a **falsifiable measurement program**, not a vibes-driven demo. Concretely: every "open-endedness" claim needs a Bedau-style neutral-shadow baseline ([Bullock & Bedau 2006](https://people.reed.edu/~mab/publications/papers/bullock.bedau.ALJ06.pdf)) and an explicit caveat per [Hickinbotham et al. (2024)](https://direct.mit.edu/artl/article/30/3/390/114972/On-the-Open-Endedness-of-Detecting-Open-Endedness); every QD claim needs QD-score, coverage, and an archive-entropy plot against three controls including pure objective and pure novelty ([Mouret & Clune 2015](https://arxiv.org/abs/1504.04909); [Lehman & Stanley](https://joellehman.com/lehman-dissertation.pdf)); every "language" claim needs topsim **and** posdis or bosdis plus held-out compositional generalization and a channel-shuffle ablation ([Chaabouni et al. 2020](https://arxiv.org/pdf/2004.09124)); every embodied/agentic claim needs a JaxLife-style compute-scaling curve with world size held constant ([Lu et al. 2024](https://arxiv.org/abs/2409.00853)); and every "physics-driven" claim needs a uniform-field counterfactual showing the inertial-microfluidics proxy is doing measurable work ([Di Carlo 2009](https://pubs.rsc.org/en/content/articlelanding/2009/lc/b912547g); [Martel & Toner](https://pmc.ncbi.nlm.nih.gov/articles/PMC4467210/)). Adopt the [EC reproducibility checklist](https://arxiv.org/html/2602.07059v1) wholesale, report medians with IQR over ≥10 seeds, and use the language guardrails in §7 to avoid the field's recurring overclaim failure mode.
