# D1 flow-head ablation

Status: **mechanism identified. The flow head, min-alignment fitness, and rank
replacement are necessary. The unit-flow normalization and the immigrants are not.**

## Purpose

The accepted claim is a 4/5 bounded pilot. This ablation asks which ingredients the
positive result actually depends on, by removing one mechanism at a time from the
same configuration, budget, and seeds.

## Protocol

Each condition runs the same study script with 5 seeds, 8 generations, 16 agents, 8
steps per generation, 14x14 world, proprioception on, displacement reward `5.0`,
fitness steps 6, eval steps 12. The gate is the accepted initial-relative gate:
evolved beats the initial population by `0.2` per held-out diagonal, and the
reversed-law alignment is at least `0.2`.

One mechanism changes per condition. The driver is
`scripts/run_d1_ablation.py`; a test checks that each condition differs from `full`
in exactly the named flag.

## Results

| Condition | Change from full | Seeds | Worst delta | Worst evolved | Worst reversed |
| --- | --- | --- | --- | --- | --- |
| `full` | none | 4/5 | +0.193 | +0.411 | +0.411 |
| `no_flow_head` | dense linear head | **0/5** | -0.730 | -0.662 | -0.662 |
| `raw_flow` | raw flow features, no unit normalization | **5/5** | +0.218 | +0.405 | +0.405 |
| `energy_fitness` | energy fitness, no min-alignment | **0/5** | -0.347 | -0.240 | -0.240 |
| `no_rank` | no rank replacement | **0/5** | -0.292 | -0.272 | -0.272 |
| `no_immigrants` | immigrant fraction 0 | **5/5** | +0.393 | +0.521 | +0.521 |

## Necessary ingredients

- **Flow-only head.** Replacing it with a dense linear head drops to 0/5 and
  `-0.730`. The four-parameter movement head is the load-bearing change.
- **Min-alignment fitness.** Replacing it with energy drops to 0/5 and `-0.347`.
  The per-direction minimum is what rules out a fixed action.
- **Rank replacement.** Removing it drops to 0/5. Without replacement, selection
  only fills dead slots and never acts on a living population.

## Not necessary in this low-dimensional regime

- **Unit-flow normalization.** Raw flow features reach 5/5, worst delta `+0.218`.
  The earlier scale diagnosis was specific to the dense genome. With a
  four-parameter head, the alignment metric is scale-invariant and the sign is what
  matters, so the raw delta is enough.
- **Immigrants.** Removing them reaches 5/5, worst delta `+0.393`. Immigrants were
  useful while exploring a dense genome, but rank replacement alone is enough with
  the small head.

## Honest reading

The `full` configuration is **not** the best on this budget: `raw_flow` and
`no_immigrants` both reach 5/5 while `full` reaches 4/5. The ablation is a
mechanism test, not a configuration search. It says the positive result depends on
the flow head, min-alignment fitness, and rank replacement. It does not say the full
flag set is optimal.

## Claim boundary

This is a bounded mechanism ablation at 8 generations. It does not upgrade the
five-seed claim. The accepted claim remains: evolved embodied organisms behaviorally
exploit the imposed law in a bounded pilot, 4 of 5 seeds, confirmed at two budgets,
under a four-parameter movement head.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_d1_ablation.py \
  --seeds 0 1 2 3 4 --generations 8 --output docs/results/d1_ablation
```

Per-condition summaries: `full.json`, `no_flow_head.json`, `raw_flow.json`,
`energy_fitness.json`, `no_rank.json`, `no_immigrants.json`. Aggregate:
`ablation.json`.
