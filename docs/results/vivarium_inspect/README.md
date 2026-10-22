# Vivarium click-to-inspect

Click a creature in the live view and read its lineage, energy, age, and body.
The selected creature gets a white ring.

## How it works

- Clicking the image maps the click to normalized world coordinates.
- `GET /inspect?x=..&y=..` finds the nearest alive agent with toroidal distance,
  records the slot and its lineage, and returns the agent fields.
- The sim loop renders a white ring on the selected slot, but only while that
  slot's lineage still matches. A revived slot cannot steal the ring.
- The page shows: `creature <slot> · lineage <id> · energy <e> · age <a> · body <r>`.

## Verified live

The browser click showed `creature 58 · lineage 58 · energy 17.8 · age 235 ·
body 1.53`.

## Endpoint

```bash
curl 'http://127.0.0.1:8765/inspect?x=0.5&y=0.5'
```

```json
{"index": 65, "x": 0.51, "y": 0.49, "lineage": 9600, "energy": 8.64,
 "age": 15, "body_radius": 1.64, "body_aspect": 1.29}
```

## Live

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/serve_vivarium.py
```

http://127.0.0.1:8765

## Tests

`tests/test_vivarium_stream.py`: the page has the inspector and `/inspect`, the
nearest-agent search wraps the seam, and the control state carries the selection.

## Next

A camera that follows the selected lineage, or a lineage history panel.
