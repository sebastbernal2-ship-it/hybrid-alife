# No-immigrant confirmation: results

Status: **5/5 seeds. The preregistered threshold was met, so the claim upgrades to
a five-seed bounded claim.**

The budget and gate were declared before launch in
`docs/CONTINUATION_HANDOFF.md` and were not changed.

## Declared configuration and budget

The `no_immigrants` ablation cell: every mechanism the ablation found necessary,
and the one it found unnecessary removed.

- Flow-only head (`w_flow`, `b_flow`), unit-flow feature, proprioception on.
- Min-alignment fitness, 6 frozen steps per direction.
- Rank replacement with `elite_fraction = 0.5`.
- Immigrant fraction `0.0`.
- Displacement reward `5.0`, eval steps `12`.
- 5 seeds, 20 generations, 16 agents, 8 steps per generation, 14x14 world.

Gate per held-out diagonal: evolved beats the initial population by at least `0.2`,
and reversed-law alignment is at least `0.2`. The absolute floor (`evolved >= 0.5`)
is a secondary diagnostic.

## Results

| Seed | Passed | Evolved minus initial | Evolved (worst direction) | Minimum reversed-law |
| --- | --- | --- | --- | --- |
| 0 | yes | +0.734 | +0.694 | +0.694 |
| 1 | yes | +0.513 | +0.551 | +0.551 |
| 2 | yes | +0.697 | +0.830 | +0.830 |
| 3 | yes | +0.516 | +0.687 | +0.687 |
| 4 | yes | +0.377 | +0.792 | +0.792 |

**5/5 seeds pass the declared gate.**

Secondary absolute floor: **5/5** pass as well, with a minimum evolved alignment of
`+0.551`.

## Law-following signature

Training alignment rises and stays high, for example seed 0:

```
0.33, 0.28, 0.68, 0.85, 0.96, 0.98, 0.84, 0.86, 0.96, 0.94,
0.96, 0.92, 0.98, 0.97, 0.95, 0.94, 0.94, 0.96, 0.95, 0.96
```

Reversed-law alignment is positive in every seed (`+0.551` to `+0.830`) and equals
the evolved alignment, so the policy re-infers the reversed flow instead of holding
a fixed action.

## Claim, upgraded

Evolved embodied organisms behaviorally exploit the imposed law:

**5 of 5 seeds, 20 generations, under a four-parameter movement head with rank
replacement and no immigrants. The initial-relative gate and the absolute floor both
pass 5/5.**

## Claim boundary

This is a bounded pilot:

- One environment: uniform imposed flow on a 14x14 toroidal world.
- Held-out diagonals, with the training cardinals in the fitness.
- Five seeds, one machine, one implementation.
- No claim about consciousness, sentience, or symbolic representation.

A second-machine replication and a new research question remain open follow-ups.

## Reproduction

```bash
JAX_PLATFORMS=cpu python scripts/run_embodied_law_study.py \
  --seeds 0 1 2 3 4 --generations 20 --population 16 \
  --steps-per-generation 8 --size 14 --eval-steps 12 --fitness-steps 6 \
  --displacement-reward 5.0 --unit-flow-features --flow-head \
  --fitness-mode min_alignment --rank-replacement --immigrant-fraction 0.0 \
  --output docs/results/embodied_law_no_immigrant_confirmation/summary.json
```
