# Thermodynamic eating (`EmbodiedConfig.thermo_efficiency`)

`thermo_efficiency` in (0, 1] closes the eating loop; 0 (default) keeps the legacy dynamics.

- Agents sharing a cell split the resource that is there. Legacy eating lets every agent draw
  its full share, so six agents in one cell are paid out more than the cell holds (documented
  by `test_legacy_eating_creates_matter_when_agents_share_a_cell`).
- Only the fraction `thermo_efficiency` of eaten mass becomes energy (at the fixed 5 energy per
  unit mass); the rest returns to the world as metabolite waste in channel 0 only, so mass is
  not duplicated across channels. The free metabolite deposit is replaced.
- Conserved quantity: `matter + energy / 5` (`tests/test_thermo_conservation.py`, including a
  25-step closed run through `step_world` with regeneration and decay off).
- Negative eat gates are clamped to zero in this mode (they would otherwise add matter).
- Validation: `EmbodiedConfig` rejects `thermo_efficiency` outside [0, 1] at construction.
- Not covered by the ledger: reward terms (`flow_alignment_reward`, `displacement_reward`),
  predation transfers, `max_energy` clipping, and Avida metabolite uptake. Costs are paid out
  of energy without a heat field, so the identity is exact only with costs at zero (as tested).
- Run: `configs/ablation_thermo.yaml`.
