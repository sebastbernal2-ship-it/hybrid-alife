# Hybrid flow head plus food gradient: smoke finding

Status: **relative gate passes, but the absolute law alignments are negative and
the absent world collapses. Not a clean law-following result.**

## What was built

- `EmbodiedConfig.food_gradient_features`: appends the local food gradient.
- `EmbodiedConfig.food_gradient_gain`: adds a fixed food-seeking reflex to the
  four-parameter flow head.
- `scripts/run_vivarium_law.py --arm hybrid`: flow head, unit-flow feature, food
  gradient, constant eat gate, min-alignment fitness, rank replacement.

## Smoke result

600 steps, 1 seed, 32 organisms, 24x24:

| Condition | Population | Alignment |
| --- | --- | --- |
| Law +x | 18.3 | -0.133 |
| Law -x | 19.0 | -0.098 |
| Law absent | 1.5 | -0.667 |

The relative gate says 1/1 passes because both law conditions beat the absent
baseline by more than `0.2`.

But the absolute alignments are **negative** in both law conditions. The population
is not following either law. It passes because the absent world is worse.

## Why

The food reflex keeps the law conditions alive, but the absent world still collapses
to one or two organisms. The survivors carry a strong fixed bias, not a law-following
response. The eat gate only feeds on a food cell, and the food reflex is not strong
enough to sustain a population without the flow.

## Honest reading

Four vivarium arms have now been tried:

- Natural: 1/3
- Dense D1 mechanisms: 1/3
- Flow head plus eat gate: not viable, absent collapses
- Hybrid: relative pass, negative absolute, absent collapses

No vivarium arm shows a robust, positive law-following response. The controlled D1
5/5 remains a special setup.

## Next options

1. Stronger survival design: a real food-seeking policy, not a fixed reflex, or
   larger food patches / lower metabolic cost so the absent world survives.
2. Accept the negative: stop the vivarium law attempts and keep the living world as
   the product, with the D1 result as the controlled research core.

## Reproduce

```bash
JAX_PLATFORMS=cpu python scripts/run_vivarium_law.py \
  --arm hybrid --seeds 0 --steps 600 --population 32 --world 24 \
  --out docs/results/vivarium_law_hybrid
```
