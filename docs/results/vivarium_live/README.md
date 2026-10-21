# Vivarium live viewer

Watch the world run in a browser, no extra dependencies.

![Vivarium live poster](poster.png)

## Run it

```bash
JAX_PLATFORMS=cpu venv/bin/python scripts/serve_vivarium.py
```

Then open **http://127.0.0.1:8765**.

Options:

```bash
python scripts/serve_vivarium.py --size 720 --fps 12 --population 128 --world 64
python scripts/serve_vivarium.py --port 9000 --warmup 60
```

## How it works

- The simulation runs in a background thread and steps the vivarium.
- The Pillow renderer draws one frame per step.
- The frame is JPEG encoded and published to a shared broadcaster.
- A small standard-library HTTP server streams those frames as
  `multipart/x-mixed-replace` MJPEG.
- The page is one `<img src="/stream.mjpg">` plus a stats bar that reads
  `/stats.json` once a second.

No Flask, no pygame, no imageio. Pillow and the standard library only.

## What you see

- Seasons, food patches, creatures with bodies and eyes, trails, corpses.
- A HUD inside the image: season, step, population, energy, births, deaths,
  alignment, corpses, mean body.
- A live stats bar under the image.

## Verification

- 3 stream tests: JPEG encoding, page references, broadcaster latest frame.
- The live server was opened in a browser and showed the running world at
  population 128, mean body 1.21, updating in real time.

## Next

The research question embedded in the world: law present versus law absent, with
the first measurement.
