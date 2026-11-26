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
- The matter identity above holds with costs at zero. For costs, rewards, predation and
  `max_energy`, use the energy ledger below. Avida metabolite uptake is outside both.
- Run: `configs/ablation_thermo.yaml`.

## Energy ledger (`EmbodiedConfig.energy_ledger`, `predation_efficiency`)

`energy_ledger: true` keeps running totals on `EmbodiedPopulationState.ledger` such that
`sum(energy) == initial + intake + external_work + injected - heat`
(`metrics/energy_ledger.py`; check it with `audit_energy_ledger(state.embodied)`).

- `external_work`: flow-alignment and displacement rewards (signed). The flow loses nothing,
  so this is energy supplied from outside the model, not a conversion.
- `injected`: newborn endowments (`initial_energy / 2`, not funded by the parent) and signed
  resets by end-of-generation selection and reseeding.
- `heat`: action, movement and basal costs, hazard damage, predation loss, `max_energy`
  clipping, reproduction cost and energy discarded from empty slots, minus the `shortfall`
  that dying agents could not pay. Hazard damage and reproduction cost are counted as
  dissipated by definition.
- `predation_efficiency` (default 1, the legacy lossless transfer): the attacker receives this
  fraction of the drained energy and the rest is heat. It works without the ledger.
- Runner metrics: `embodied_dissipated_heat`, and `avida_dissipated_heat` when the Avida
  energy economy is on. The two totals are separate ledgers. There is no spatial heat field.
- Dynamics change in one place: the legacy reproduction scatter lets unused entries rewrite
  slot 0, so a parent in slot 0 pays nothing. With the ledger on those writes are dropped.
- `tests/test_energy_ledger.py` checks the balance with every term enabled through
  `step_world`, reproduction and selection, and checks it against the world's matter arrays.
- Totals are float32: expect rounding residuals that grow with run length.
