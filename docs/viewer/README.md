# Immersed-boundary swimmer viewer (prototype v2)

A WebGL2 viewer for one recorded immersed-boundary run
([method and its limits](../extensions/immersed_boundary.md)): five flexible swimmers with
a prescribed gait in a channel with a post. A visual prototype awaiting approval, not
evidence for any research claim. The swimmers are not evolved agents.

## Run

    make -C native
    export PYTHONPATH=src JAX_PLATFORMS=cpu
    python scripts/run_viewer.py export       # about 6 minutes on one core
    python scripts/run_viewer.py serve        # http://127.0.0.1:8765/
    python scripts/run_viewer.py standalone --out replay.html   # double-clickable, offline

Controls: drag to pan, scroll to zoom, click a swimmer, Space to pause, arrows to step,
`0` to reset. URL options: `beat`, `paused`, `field=speed`, `tracers=0`, `zoom`, `cx`,
`cy`, `select`.

## Scene

384 x 224 cells, nu = 1/6, periodic left to right, walls top and bottom, one post. Fluid
at rest, no background flow, no chemical signal. Swimmers are 1.5 times the default
`ib.swimmer`, wave amplitude 3 before scaling, 2000 steps per beat, soft contact
(range 3, k 0.1) and soft walls (range 4.5, k 0.1). 2 ramp beats, then 90 recorded beats
at 8 records per beat.

## What is drawn

| Visual | Source |
|---|---|
| Vorticity or speed | fluid velocity, block-averaged 2x2, cubic B-spline in the shader, compressed colour scale |
| Swimmers | every Lagrangian point; head = ring grown 0.6 cell, tail ribbon 2.3 to 0.7 cells wide |
| Walls, post | the analytic surfaces used by the kernel |
| Tracers | massless points integrated in the browser (RK2); trails sample once per beat over 8 beats |
| Between records | Catmull-Rom for points, linear for velocity |

## Overlap and penetration, this scene

Measured on the drawn shapes at all 721 records: 0 frames with overlapping swimmers
(smallest gap 1.53 cells), 0 with crossing outlines, 0 with a wall or post entered
(smallest gap 3.56 cells). Over every time step, the closest points of two swimmers were
2.977 cells apart (contact range 3) and the smallest point-to-surface gap was 4.452
(wall range 4.5). The penalties barely engaged, so this scene does not test them under
strong forcing; `tests/test_ib_walls.py` covers a forced wall case. Both are soft
penalties, not hard constraints.

## Limits

- Swim speeds are not grid-converged (expect them low; see the IB document), the beat
  period inflates stride by a few percent, and the Reynolds number is not bacterial. 2-D.
- Motion is slow: 18 to 38 cells of net displacement in 90 beats.
- Vorticity within about 2 cells of a body is a property of the smoothed kernel.
- The replay is about 24 MB (about 33 MB as standalone HTML).

## Frame rate

See `FPS_LINE`.
