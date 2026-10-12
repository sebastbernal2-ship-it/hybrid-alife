# Resource-coupled active-probe pilot v1

This archive repeats the three-seed active-probe pilot after enabling flow advection of resources in field-discovery configs.

- Seeds: `0`, `1`, and `2`.
- Training: two generations per condition.
- Evaluation: eight steps per condition.
- Resource advection strength: `0.25`.

## Finding

Flow-to-resource coupling increased the counterfactual flow-information diagnostic from near zero to positive values in the field conditions.
The probe rate still stayed near `0.50` in every condition.
Survival remained `1.0` and reproduction remained `0`.

This separates two issues:

- The environment now contains a clearer causal route from hidden flow to resource transport.
- The current evolutionary horizon still does not produce selective active probing.

This is engineering evidence only.
It does not establish environmental model-use or consciousness.
