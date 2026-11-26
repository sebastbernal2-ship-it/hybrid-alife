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
- By default the signal is an intensity (gate x `signal_deposit`) in channel 1.
- Content (opt-in): set `WorldConfig.concentration_channels` to `1 + K` and
  `EmbodiedConfig.signal_content: true`. Channel `k` (1 to K) receives
  `gate * signal_deposit * message[k-1]`; needs `message_size >= K`. Diffusion, decay and
  advection are linear and identical across channels, so one emitter's message reaches a
  receiver scaled by one common non-negative factor: direction (signs and ratios) is
  decodable, absolute magnitude is not. Overlapping emissions add, and the channels carry
  signed values, which is an abstraction rather than a physical concentration.
- More channels widen the patch and sixth sense, so controller genomes are not
  interchangeable with two-channel checkpoints. `SIXTH_SENSE_WIDTHS` describes the default
  two-channel layout. Avida's coupling deposit (Python and native) goes into every
  concentration channel, so keep `AvidaConfig.metabolite_deposit` at 0 when decoding
  content. The `mean_concentration` metric averages over all channels.
- The emitter senses its own deposit. With
  `concentration_decay == 1` the signal accumulates without bound.
- Validation: `initialize_embodied_population` rejects a negative `signal_deposit`,
  `use_comms=False`, `blind`, and `concentration` in `hidden_field_groups` (the law-study
  config hides concentration), because the signal would be silently unobservable.
- Run: `configs/ablation_chemical_signaling.yaml`.
