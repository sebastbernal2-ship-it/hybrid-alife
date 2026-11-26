# Live-LBM flow viewer (prototype)

A WebGL2 viewer for recorded `lbm_live` runs with dipole coupling. This is a visual
prototype awaiting approval. It is not evidence for any research claim: controllers are
random initial genomes, mutated at reproduction, and untrained.

## Run

    make -C native
    export PYTHONPATH=src JAX_PLATFORMS=cpu
    python scripts/run_viewer.py export --scene obstacle      # or --scene periodic
    python scripts/run_viewer.py serve                         # http://127.0.0.1:8765/
    python scripts/run_viewer.py standalone --out viewer.html  # double-clickable, offline

Controls: drag to pan, scroll to zoom, click to inspect, Space to pause, arrow keys to
step, `0` to reset. URL options: `t`, `paused`, `field=speed`, `signal=0`, `particles`, `trail`.

## What maps to data

| Visual | Source |
|---|---|
| Vorticity or speed | live LBM velocity, 256x256, every 3rd sim step, lattice units |
| Cylinder and walls | the LBM solid mask |
| Signal haze | `world.concentration[..., 1]`, square-root scale, 9-tap display smoothing |
| Tracers and trails | viewer-side GPU particles advected through the recorded velocity (`lbm_substeps` LBM steps per sim step); random respawn, not simulated particles |
| Position, energy, age, lineage, genes | per-step agent state; founder follows `parent_id` |
| Heading and inspector swim value | commanded swim displacement (action x 0.05) |
| Body proportions and size | `body_radius` and `body_aspect` genes |
| Color | founder lineage; the six most present, all others gray |

## Illustrative only (not simulated)

- Flagella. Only the beat rate is tied to swim command magnitude.
- Body drawing scale, the 2.2x base elongation (genes alone span nearly round shapes),
  the egg taper, shading and shadows.
- Motion between records: Catmull-Rom for positions, linear in time for fields.
- Birth grow-in and death fade (the events are real, their animation is not).

## Differences from the stock simulator config

`export.viewer_config` is a visualization scene, not a study config: 256x256 grid,
viscosity 0.008, cylinder radius N/12, parabolic initial profile with a weak perturbation
and 30000 warm-up steps, 60 LBM substeps, drag 0.007, dipole length 6, toroidal agent
world. Passive agent drift and signal advection are set to match the fluid displacement
per sim step. Commanded swimming (up to 12.8 cells per step) is much faster than the flow.

## Limits

- No solid collision: organisms pass through the cylinder and walls (drawn faded inside).
  The toroidal agent world also wraps across the channel walls.
- Dipole forces are point forces on lattice cells, held for one sim step, so they show as
  bead-like vorticity chains behind swimmers.
- A founder color is not unique past six lineages; click a legend entry to isolate one.
  The six colors pass the adjacent-pair check, not the all-pairs color-vision check.
- The replay is recorded, about 13 MB for 240 steps (18 MB standalone HTML).
- A long frame stall slows playback instead of skipping.

## Frame rate

Measured on the obstacle replay:

- NVIDIA RTX 5080 Laptop GPU (headless Windows Chrome, D3D11), 1106x750 canvas,
  playing with 9000 tracers: 239.9 fps mean, 4.3 ms p95. This sits at the 240 Hz frame
  pacing cap, so it shows headroom over 60 fps rather than a maximum.
- Software GL (SwiftShader, 3 CPU cores) is far below real time: about 160 ms per frame
  even at 720x450 with 2000 tracers. The GIF was stepped frame by frame, so its smoothness
  does not reflect software frame rate.

Linux headless Chromium inside WSL fell back to SwiftShader.
