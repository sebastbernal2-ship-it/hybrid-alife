# Vivarium viewer controls

The live page now has controls, not just a stream.

## Controls

- **Pause / Resume**: freeze the simulation and inspect the world.
- **Speed**: 0.25x to 4x.
- **Reset**: start a fresh world with the same configuration.

## Endpoints

- `GET /control?action=toggle`
- `GET /control?action=pause`
- `GET /control?action=resume`
- `GET /control?action=reset`
- `GET /control?speed=2`

The sim thread reads the controls each loop: it pauses, scales the frame interval,
or re-initialises the state on reset.

## Verified

- Pause froze the step counter, resume continued.
- Speed 2 was accepted.
- Reset restarted the world at a low step and a fresh population.

## Live

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/serve_vivarium.py
```

http://127.0.0.1:8765

## Tests

`tests/test_vivarium_stream.py`: the page has the control ids and `/control`, and
the `Controls` object toggles pause, clamps speed, and consumes reset.

## Next

The viewer could add click-to-inspect on an organism, or a camera that follows a
lineage.
