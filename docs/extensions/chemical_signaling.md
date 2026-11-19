# Chemical signaling (`EmbodiedConfig.chemical_signaling`)

Messages travel as a diffusing chemical signal instead of an instantaneous same-cell broadcast.

- An emission (gate above `message_gate_threshold`) deposits `signal_deposit` into
  `concentration[..., 1]` at the emitter's cell. The world's concentration diffusion, decay and
  flow advection set signal speed, range and lifetime. Terraforming and the Avida VM use
  channel 0, so channel 1 is free.
- The legacy "neighbour message energy" term of the crowding sense is zeroed in this mode.
  Agents read the signal through the concentration channels of their sixth sense and patch.
- `tests/test_diffusive_signaling.py`: the signal never outruns one cell per step (discrete
  light cone), mass is conserved with decay 1, and a neighbour one cell away senses the signal
  one step after emission (and does not when concentration is hidden).
- The signal is an intensity (gate x `signal_deposit`), not message content, because the world
  has two concentration channels. The emitter senses its own deposit. With
  `concentration_decay == 1` the signal accumulates without bound.
- Validation: `initialize_embodied_population` rejects a negative `signal_deposit`,
  `use_comms=False`, `blind`, and `concentration` in `hidden_field_groups` (the law-study
  config hides concentration), because the signal would be silently unobservable.
- Run: `configs/ablation_chemical_signaling.yaml`.
