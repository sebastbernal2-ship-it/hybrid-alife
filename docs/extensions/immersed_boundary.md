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
`tests/test_immersed_boundary.py` repeat the coarse cases. The sheet and swimmer
refinement rows come from `scripts/run_ib_convergence.py` (raw JSON in
`docs/results/ib_convergence/raw`, fits in `summary.json`) and are not rechecked by CI.
The cylinder table and the other single-run numbers were run once with throwaway drivers.

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

Speed error against theory (cells per wavelength; box height H; 8 beats, the first two
dropped; +/- is the standard error over the other six, which are not independent):

| cells | H = 1 wavelength | H = 2 wavelengths |
|---|---|---|
| 32 | -29.8 +/- 0.04 % | -25.6 +/- 0.04 % |
| 48 | -22.2 +/- 0.2 % | -18.0 +/- 0.2 % |
| 64 | -17.3 +/- 1.0 % | -12.7 +/- 0.8 % |
| 96 | -13.5 +/- 0.2 % | -9.0 +/- 0.2 % |
| 128 | -11.2 +/- 0.1 % | -6.6 +/- 0.1 % |

- **H = 2 is consistent with first order toward Taylor's value.** The observed order of
  the error is 0.98 from 32 to 128, 1.07 +/- 0.10 from 96 to 128, and 0.9 to 1.2
  (+/- 0.2) for the pairs through 64. A first-order line through 96 and 128 gives +0.5 %.
  This is still an extrapolation; the closest run is 6.6 % low.
- **H = 1 does not tend to the same value:** the same line gives -4.3 %. The theory is
  for an unbounded fluid and the periodic images are not negligible at H = 1.
- Longitudinal material motion of the elastic sheet is not the cause: its measured
  leading-order effect on the speed is below 0.03 % in every run.
- The 64-cell speed varies by about 10 % from beat to beat (amplitude 1.05 of
  prescribed). The cause is not known. Averaging three beats, as before, was too short there.
- A single earlier set of runs also found the error did not come from inertia, `bk` or
  stiffness. A 192-cell run is unstable: its `bend_k = 12` is past the explicit limit.

**At the resolutions a swimmer would use, expect speeds 15 to 30 % low.**

### Free swimmer: not grid-converged

Refinement by `r`: all lengths times `r` (including the box, 64r x 96r), period `2000 r^2`
(fixed wave Reynolds number at fixed `nu`), point spacing 1 cell. To keep the physical
stiffness fixed, **`bend_k = 0.2 r` and `stretch_k = 0.05 / r`**. In 2-D per unit depth a
force scales as `mu U ~ 1/r`, so `B = force x length^2 ~ r` and the joint constant is
`B/ds` with `ds` one cell; `EA ~ 1/r` likewise. An earlier version of this page gave
`bend_k ~ r/period` and `stretch_k ~ 1/(r period)`. Read literally that is softer by
`1/r^2` and gives stride 0.103 at r = 2 and a realised amplitude that falls steadily; it
is not what produced the old table, which the formulas above reproduce to within 0.002.

Stride is per beat in r = 1 cells, mean from beat 3 on (beats 3 to 6; 3 and 4 at r = 6).
Tip amplitude is the first harmonic of the tail tip's offset from the head axis, over `r`.

| r | stride | beat-to-beat s.d. | tip amplitude |
|---|---|---|---|
| 1 | 0.2119 | 0.004 | 1.91 |
| 1.5 | 0.2999 | 0.002 | 1.78 |
| 2 | 0.3509 | 0.001 | 1.71 |
| 3 | 0.4057 | 0.0003 | 1.62 |
| 4 | 0.4352 | 0.0007 | 1.57 |
| 6 | 0.4681 | 0.0004 (2 beats) | 1.52 |

**The swimmer is not converged at any resolution run, and the order is not pinned down.**

| fit | observed order | limit |
|---|---|---|
| `L - c r^-p`, all 20 triples of the six rows | 0.59 to 0.79 (0.68 to 0.77 for triples with r = 6) | 0.55 to 0.63 (0.556 to 0.573 with r = 6) |
| `L - a/r - b/r^2`, triples from r >= 1.5 | 1 assumed | 0.527 to 0.544 |
| `L - a/r`, r = 4 and 6 | 1 assumed | 0.534 |

The apparent order rose with resolution up to r = 4 (0.79 from r = 2, 3, 4), then the
triple 3, 4, 6 gave 0.68. So asymptotic first order is plausible but **not shown**. Both
fit forms through r = 2, 3, 4 predicted 0.466 to 0.467 at r = 6, against 0.468 measured. Taking
0.53 to 0.57 as the range of the limit, r = 6 is 12 to 18 % low, r = 4 is 18 to 24 % low
and r = 1 is 60 to 63 % low. These ranges are spreads between fit forms, not statistical
confidence intervals.

What was checked, changing one thing at a time (stride; baseline r = 1: 0.212, r = 2: 0.351):

| change | r = 1 | r = 2 | reading |
|---|---|---|---|
| 10 beats | 0.213 | 0.351 | averaging window: negligible at r >= 2, about 2 % scatter at r = 1 |
| box x1.5 / x2 | 0.201 / 0.197 | 0.343 / 0.340 | periodic images add 3 % (r = 2) to 8 % (r = 1) |
| period x4 | 0.207 | 0.332 | inertia: a few percent |
| point spacing 0.5 cell | 0.20 (erratic) | 0.351 | stride unchanged at r = 2, though tip amplitude rises to 1.95 |
| stiffness x0.5 / x4 | 0.120 / 0.471 | 0.209 / 0.702 | the gait is stiffness limited: scaling errors matter a lot |

Compressibility is not it: the largest density range falls from 0.0022 (r = 1) to
0.0002 (r = 6). The remaining resolution error behaves like a boundary error whose size
is set in cells: the kernel support makes thin bodies hydrodynamically thicker than drawn
(see the cylinder), and a narrower kernel moves each coarse result toward the fine ones.
That is an interpretation consistent with the data, not a proof; the slower-than-first
apparent order and the drift of the realised amplitude are not explained.

**The default swimmer (r = 1: head 8 x 5 cells, tail 20 cells, wave amplitude 1.5) is
not grid-converged. Do not quote its speed.** The qualitative behaviour (direction, no
motion without actuation) holds. A standing-wave actuation still drifts, because a
flexible tail does not follow a reciprocal rest shape reciprocally; this is therefore not
a scallop-theorem test, and no rigid-kinematics scallop test was done.

### Narrower kernels (opt-in, limited use)

`Structure.from_bodies(..., kernel=3)` uses the Roma, Peskin and Berger 3-point delta in
the time loop and `kernel=2` the linear hat; `spread()` stays 4-point. The default is 4
and is unchanged. Both conserve total force and first moment, and both lower cylinder drag.

| r | 4-point | 3-point | 2-point |
|---|---|---|---|
| 1 | 0.212 | 0.271 | 0.303 (noisier) |
| 1.5 | 0.300 | 0.344 | 0.369 (noisier) |
| 2 | 0.351 | 0.386 | unstable |
| 3 | 0.406 | 0.431 | unstable |
| 4 | 0.435 | 0.456 | unstable |

For this swimmer the 3-point kernel at `r` is roughly the 4-point one at 1.3 `r`. It does
**not** change the order (0.72 from r = 2, 3, 4; limit 0.564, or 0.541 with the two-term
form, close to the 4-point values). **It is not a general fix:** on the sheet it ran only
at 32 cells (-20.7 % against -25.6 %) and went unstable at 48 to 128; the 2-point kernel
failed every sheet run. Smaller stiffness or spacing for those cases was not explored.
Do not use `kernel=2` or `3` without checking stability and resolution.

Reproduce (finished cases are skipped, so delete a raw file to rerun it; r = 6 takes
over an hour):

```
for s in swimmer controls sheet kernel; do
  python scripts/run_ib_convergence.py --study $s --jobs 4 --output /tmp/ib_raw
done   # sheet includes 192 cells, which is recorded as failed (unstable)
python scripts/analyze_ib_convergence.py   # reads docs/results/ib_convergence/raw
```

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

`make sanitize-build sanitize-test` (also a CI job) runs the native tests, including the
narrow kernels, under AddressSanitizer and UndefinedBehaviorSanitizer; restore the normal
build with `make -B -C native`.

## Not done

- A grid-converged free swimmer, or an established asymptotic order for it.
- IB accuracy away from `nu = 1/6`. TRT or MRT collision, implicit coupling and
  sharp-interface schemes were not tried. Multi-direct forcing was not tried either: it
  reduces slip in direct-forcing schemes, whereas here points already move with the
  interpolated velocity.
- Wall contact, self-contact, hard non-overlap constraint, scalar transport, Brownian
  motion, 3-D, threading, and any coupling to agents or evolution.
