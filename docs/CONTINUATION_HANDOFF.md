# Continuation Handoff — hybrid-alife

> Purpose: enable a brand-new Claude (or human) chat starting from zero context to
> continue this project without re-deriving state. Read this first, then `README.md`,
> then `docs/scientific_validation.md`.

## Pointers

- **Repo URL:** https://github.com/sebastbernal2-ship-it/hybrid-alife
- **Main branch tip before this sprint:** `0c25070bc0c29d82371a68e1e150704c5ddbded1`
  (`merge: sprint/poet-transfer-fast into final integration`)
- **This sprint branch:** `sprint/continuation-handoff-fast` (docs-only).
- **Date of handoff:** 2026-05-11.

## TL;DR project state

Hybrid artificial-life simulator in JAX. Two coupled branches (embodied GRU/MLP agents +
Avida-style instruction VM) share a 2D world of proxy microfluidic fields (curvature,
shear, lift, enrichment, concentration, metabolites). Evolution loop with tournament,
novelty archive, MAP-Elites. Scientific-depth metrics (Bedau, QD, topsim/posdis/bosdis,
Hill 1D lineage) are implemented and unit-tested. A v1 transfer matrix and a compute-
scaling harness exist as scripts but POET-style coevolution is **not** yet wired in.

## Known validation (snapshot at handoff)

- `pytest -q` → **104 tests passed in ~83 s** on CPU.
- QD-active smoke (`configs/qd_active.yaml`) at gen 5:
  - coverage `0.4375`, qd_score `77.9649`, novelty archive size `72`.
- Communication benchmark acceptance:
  - compositional protocol topsim / posdis / bosdis all `1.0`.
- Compute-scaling, budget 6, gen 5:
  - coverage `0.5`, qd_score `15.4245`, novelty archive size `36`.
- Transfer/scaling artifacts exist under `outputs/` from prior sprints.
