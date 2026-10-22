# Flow head plus eat gate: design not viable as a survival test

Status: **design finding, not a completed comparison.**

## What was built

`EmbodiedConfig.flow_head_eat_gate` gives the four-parameter flow head a constant
eat gate, so a movement-only brain can eat when it is standing on food.
`scripts/run_vivarium_law.py --arm d1_flow` uses it with the unit-flow feature,
min-alignment fitness, and rank replacement.

## Smoke result

600 steps, 1 seed, 32 organisms, 24x24 world:

| Condition | Population | Alignment |
| --- | --- | --- |
| Law +x | 19 | -0.228 |
| Law -x | 19 | -0.228 |
| Law absent | 5 | undefined |

The law conditions survive, but both give the same alignment, a fixed -x bias.
The absent condition collapses, so there is no valid baseline for the gate.

## Why it fails as a comparison

The flow head has no food-seeking. The eat gate only feeds it when it happens to
stand on a food cell. With a law, the flow carries it through patches. Without a
law, it starves. So the absent control cannot be measured.

This is not evidence about law-following. It is evidence that a movement-only brain
cannot survive in this world without a food-seeking behavior.

## What this means

The low-dimensional D1 mechanism cannot be dropped into the living world as-is.
The controlled D1 5/5 claim used a displacement reward that fed the organism and no
requirement to eat. The vivarium requires both survival and movement.

## Next options

1. **Hybrid movement head**: flow-following plus a simple food gradient, so the
   organism can survive in both law conditions and the absent control.
2. **Natural controller with the unit-flow input** (the dense arm) already failed
   at 1/3.
3. **Accept the negative**: the living-world law effect is not demonstrated, and
   the controlled D1 result remains a special setup.

The honest current status: no vivarium arm shows a robust law-following response.
