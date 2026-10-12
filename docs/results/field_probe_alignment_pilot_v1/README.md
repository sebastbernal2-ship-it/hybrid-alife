# Flow-alignment active-probe pilot v1

This archive repeats the resource-coupled active-probe pilot with a bounded task payoff for moving with local hidden flow.

- Seeds: `0`, `1`, and `2`.
- Training: ten generations per condition.
- Evaluation: eight steps per condition.
- Resource advection strength: `0.25`.
- Flow-alignment reward: `0.5`.

## Finding

The task payoff increased post-intervention energy in field conditions relative to the no-field control.
The probe rate still stayed near `0.50` in every condition.
Indirect and intervention probe-outcome information remained similar to direct information.
Survival remained `1.0` and reproduction remained `0`.

The longer horizon and explicit task payoff therefore did not produce evidence of selective active probing.
The next experiment needs a task with non-saturated fitness and a held-out prediction or choice score, rather than more generations under the current saturated population.

This is engineering evidence only.
It does not establish environmental model-use or consciousness.
