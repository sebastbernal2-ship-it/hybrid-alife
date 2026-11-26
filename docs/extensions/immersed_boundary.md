# Immersed-boundary bodies and contact (opt-in)

Swimmers as extended elastic bodies in the D2Q9 live solver. Nothing in the world
simulator or evolution calls this code; you use it only by importing
`hybrid_alife.physics.ib`. `lbm.cpp` and every legacy default are unchanged.

| File | Role |
|---|---|
| `native/ib.cpp` | Elastic and actuation forces, delta-kernel spreading and interpolation, time loop (includes `lbm.cpp` and uses its `step_once`) |
| `native/contact.cpp` | Spatial-hash contact between points of different bodies |
| `src/hybrid_alife/physics/ib.py` | Body builders (`swimmer`, `cylinder`, `sheet`), `ImmersedSim`, state export |
| `src/hybrid_alife/physics/units.py` | Lattice to SI mapping |
| `scripts/run_ib_smoke.py` | Two swimmers with contact, writes a state file |
| `benchmarks/bench_immersed_boundary.py` | Throughput |

## Method

Peskin's immersed boundary (Acta Numerica 11:479, 2002) with the 4-point kernel and Guo
forcing. Per step: point forces at X^n, spread to the grid, one fluid step, interpolate
`u = (sum f c + F/2)/rho` at X^n, `X^{n+1} = X^n + u`. The coupling is explicit.

A body is points joined by stretch springs and bend springs on signed joint angles.
The cell body is a hub-and-ring truss; the tail is a chain whose rest angles follow a
travelling wave `A sin(omega t - k s)`. Only the rest shape is prescribed: the tail is
flexible and the realised shape comes from the balance with the fluid.

**Force- and torque-free by construction.** Every internal force is the gradient of an
energy that does not change under rigid translation or rotation, so its sum and torque
are zero for any actuation. There is no thrust force. The 4-point kernel keeps total
force and first moment, so spreading adds neither. Measured on the swimmer: net force
below 4e-16 of the summed force magnitude, net torque below 3e-15, total fluid momentum
constant to 1e-10. Limits of this claim: D2Q9 BGK on a periodic grid does not conserve
fluid angular momentum exactly, and the torque diagnostic uses unwrapped coordinates, so
it means nothing for periodic filaments or contact across the box edge. Tethers are
external forces and exist for the drag benchmark. The box has no wall to push against,
but periodic images of a body do interact with it.

Contact is a soft pair force `k (r_c - r)` along the line of centres, between points of
different bodies, found with a cell list (checked against brute force). It conserves
momentum (and torque when no pair wraps the box). **It is a penalty, not a hard
constraint.** In the head-on test with `r_c = 3`, the closest approach was 1.46 cells
without contact and 2.53 with `k = 0.3` (1.65 at 0.03, 2.11 at 0.1). The ring outlines
never crossed. That is one tested scenario, not a proof: with point spacing near or above
`r_c`, or weak stiffness under strong forcing, bodies can pass between points. The IB
velocity field is continuous and by itself already resists crossing; at `r_c = 1.5`
contact made almost no difference in this test. There is no self-contact.

Walls: `mask` and `ImmersedSim.from_live(structure, live_lbm)` work, but a point whose
4x4 stencil touches a solid node raises (no wall contact model). `from_live` keeps the
channel body force, not the periodic box's mean-flow relaxation.

## Validation

All numbers are lattice units, `nu = 1/6` unless stated. Tests in
`tests/test_immersed_boundary.py` repeat the coarse cases; the finer rows were run once
with throwaway drivers built on the public API and are not rechecked by CI.

### Drag on a square array of cylinders

Reference: Sangani and Acrivos, Int. J. Multiphase Flow 8:193 (1982),
`F/(mu U) = 4 pi / (-ln(c)/2 - 0.738 + c - 0.887 c^2 + 2.038 c^3)`, with `c` the area
fraction and `U` the box-averaged velocity. The kernel radius is the nominal one.

| box | c = 0.05 error | c = 0.10 error |
|---|---|---|
| 32 | +18.6 % | +17.7 % |
| 64 | +8.3 % | +7.9 % |
| 128 | +3.5 % | +3.6 % |

First-order convergence. At 64, c = 0.05, the error equals an effective radius about
0.5 cell larger than nominal: **thin IB bodies are hydrodynamically fatter than drawn.**
Tether force matched the applied body force to within 5e-5. The existing bounce-back
solver on the same arrays (compared as `F = g L^2`) is within 0.1 to 1.1 %, except
8.0 % at box 32, c = 0.1.

### Taylor's swimming sheet

Reference: Taylor, Proc. R. Soc. A 209:447 (1951), `U/c = (bk)^2/2 (1 - 19/16 (bk)^2)`,
times Tuck's finite-inertia factor `(1 + F)/(2F)`,
`F = sqrt((1 + sqrt(1 + R^2))/2)`, `R = omega/(nu k^2)` (J. Fluid Mech. 31:305, 1968;
that factor is derived at leading order for transverse motion and is applied here to the
inextensible formula). `b` is the measured amplitude, not the prescribed one; `U` is
relative to the fluid half a box away. `bk = 0.2`, `R = 0.24`. The sheet swims against
the wave, as it should.

Speed error against theory (cells per wavelength; box height H):

| cells | H = 1 wavelength | H = 2 wavelengths |
|---|---|---|
| 32 | -30 % | not run |
| 48 | -22 % | -18 % |
| 64 | -20 % | -15 % |
| 96 | -14 % | -9.0 % |
| 128 | -11 % | -6.7 % |

In the taller box a first-order fit through 96 and 128 (or 48 and 128) extrapolates to
within 1 % of theory. That is an extrapolation; no run itself is closer than 6.7 %.
A 192-cell run went unstable because its `bend_k = 12` is past the explicit limit below.

The error does not come from inertia (at 48 cells, `R = 0.06`: -21.6 %), from `bk`
(0.1: -23 %; 0.4: -18 %), or from stiffness (-22 %). It falls with resolution and with
box height. **At the resolutions a swimmer would use, expect speeds 15 to 30 % low.**
The 64-cell realised amplitude differs from the others (1.04 of prescribed), so that row
is a less clean point in the sequence.

### Free swimmer: resolution is the main error

Same physical swimmer at fixed elastoviscous number (`bend_k` proportional to
`r/period`, `stretch_k` to `1/(r period)`), with `R` fixed, geometry and box times `r`,
period times `r^2`, point spacing 1 cell. Stride is per beat in `r = 1` cells.

| r | stride | against the extrapolated 0.52 |
|---|---|---|
| 1 | 0.214 | -59 % |
| 2 | 0.351 | -33 % |
| 3 | 0.406 | -22 % |
| 4 (2 beats averaged, others 4) | 0.435 | -17 % |

The differences shrink (0.137, 0.055, 0.029), consistent with first order in cell size.
The limit 0.52 is a first-order extrapolation from r = 3 and 4, not a computed value.

**The default swimmer (r = 1: head 8 x 5 cells, tail 20 cells, wave amplitude 1.5) is
not grid-converged. Do not quote its speed.** The qualitative behaviour (direction, no
motion without actuation) holds. A standing-wave actuation still drifts, because a
flexible tail does not follow a reciprocal rest shape reciprocally; this is therefore not
a scallop-theorem test, and no rigid-kinematics scallop test was done.

### Viscosity and kinetic stability

Fluid alone, Taylor-Green decay on 128^2, error in the measured viscosity:
0.06 % at `nu = 0.02`, 0.0002 % at 1/6, 0.7 % at 1. IB accuracy (wall slip) was only
measured at `nu = 1/6`.

Explicit-coupling limits, default swimmer, 64 x 96, 6000 steps, ramp 1000:

| parameter | stable | unstable (raises) |
|---|---|---|
| `stretch_k` (`bend_k` 0.2) | up to 1.0 | 1.5 and above |
| `bend_k` (`stretch_k` 0.05) | up to 6 | 8 |
| `stretch_k` at `nu` 0.02 and 1.0 | up to 0.8 | 1.5 and above |
| `nu` | 0.002 to 4 all ran | none found |
| beat period | 50 to 2000 all ran | none found |
| cylinder `tether_k` (3000 steps) | 0.05 to 1.6 all ran | none found |

"Ran" means finite and below speed 0.3, not accurate. The kernel returns an error and
leaves the state untouched if speed passes 0.3 or the state is non-finite.

## Physical units

`dt = nu_lu dx^2 / nu_water`. With `dx = 0.25 um`, `nu_lu = 1/6`: `dt = 10.4 ns`. The
default swimmer is then a 2 x 1.25 um body with a 5 um tail, about E. coli sized.
Bacterial values are from H. C. Berg, *E. coli in Motion* (2004): speed about
25 um/s, flagellar bundle about 100 Hz. Force per unit depth converts with
`rho dx^3 / dt^2`.

| number | E. coli in water | this code |
|---|---|---|
| Re = UL/nu (L = 2 um) | 5e-5 | exact in the mapping; see below for runs |
| wave Reynolds `omega/(nu k^2)`, wavelength 5 um, 100 Hz | 4e-4 | 4e-4 at 960,000 steps per beat; 0.19 at the default 2000 |
| Pe, small molecule (D = 1e-9 m^2/s) | 0.05 | not simulated |
| Pe, protein (D = 1e-10 m^2/s) | 0.5 | not simulated |

- **Matched time step** is affordable only for a few beats: 960,000 steps per beat on a
  64 x 96 box is about 6 to 13 minutes per beat on one core.
- **Accelerated beats** (period 2000, 480 times faster, with stiffness scaled as
  above) are the practical regime. Measured stride per beat against period at `r = 1`:
  2000: 0.2141, 8000: 0.2090, 16000: 0.2068, 32000: 0.2053, 64000: 0.2043. The 64000
  run has wave Reynolds 0.006, still 15 times the bacterial value. By extrapolation the
  Stokes value is about 0.202 to 0.203, so **period 2000 overstates the stride by about
  5 to 6 % at r = 1**. This inertia error is much smaller than, and separate from, the resolution
  error.
- At period 2000 the lattice Re on the 28-cell total length is about 0.02, not 5e-5.
- **No scalar is transported**, so no Peclet regime is simulated or validated; the Pe
  rows only say what a future transport solver would have to match.
- **2-D only.** A rotating helical flagellum, counter-rotation of the body, and 3-D
  drag cannot be represented. The planar wave is closer to a sperm cell than to
  *E. coli*. 2-D Stokes flow also depends logarithmically on box size. SI speeds from
  this model are not predictions of bacterial speeds.
- No Brownian motion, which matters for real bacteria.

## State export for viewers

`ib.record_states(sim, path, frames, steps_per_frame, dx_m=None, dt_s=None)` writes
UTF-8 JSON Lines. Line 1:

```json
{"schema": "hybrid-alife/ib-state/v1", "nx": 128, "ny": 80, "dx_m": 2.5e-07,
 "dt_s": 1.04e-08,
 "bodies": [{"kind": "swimmer", "start": 0, "stop": 43, "edges": [[0, 1]]}]}
```

`edges` index into the body's own `[start, stop)` range. Every later line is
`{"step": 200, "points": [[x, y], ...]}` with all points in header order. There are
`frames + 1` of them, the first being the initial state. Coordinates are lattice cells in
the LBM frame (x along `nx`; this is world axis 0 when coupled to `LiveLBM`), rounded to
1e-5 and **unwrapped**: take them modulo `nx`, `ny` to draw, and do not draw an edge
the long way across the box. The fluid field is not exported. For a periodic `sheet`,
the closing edge joins its last and first points across the box.

```
python scripts/run_ib_smoke.py --output outputs/ib_smoke.jsonl
```

## Performance

Intel Core Ultra 9 275HX under WSL2, g++ 13 `-O3`, one thread, best of 3. This run
shared the machine with one other pinned job; with three busy, rates were about half
(about 8 million lattice updates per second, MLUPS). Treat as indicative.

| grid | bodies (points) | steps/s | MLUPS |
|---|---|---|---|
| 64 x 96 | live solver, none | 3197 | 19.6 |
| 64 x 96 | 1 (43) | 2622 | 16.1 |
| 128 x 128 | 4 (172), contact | 1067 | 17.5 |
| 256 x 256 | live solver, none | 274 | 18.0 |
| 256 x 256 | 16 (688), no contact | 230 | 15.0 |
| 256 x 256 | 16 (688), contact | 234 | 15.3 |
| 256 x 256 | 64 (2752), contact | 180 | 11.8 |

The fluid step dominates. The kernel is single threaded and is not optimised.

## Sanitizers

`libib.so` and `libcontact.so` were built with
`-fsanitize=address,undefined -fno-sanitize-recover=all`, and
`tests/test_immersed_boundary.py` passed with ASan preloaded and no reports (leak check
off, as CPython leaks by design). A deliberately undersized `ib_spread` buffer produced a
heap-buffer-overflow report, so the instrumentation was live. This is a manual run:
this change does not add it to CI.

## Not done

- A free-swimmer run close to its grid limit (r = 4 is still about 17 % below the
  extrapolation).
- IB accuracy away from `nu = 1/6`; no multiple-relaxation-time or direct-forcing variant.
- Wall contact, self-contact, hard non-overlap constraint, scalar transport, Brownian
  motion, 3-D, threading, and any coupling to agents or evolution.
