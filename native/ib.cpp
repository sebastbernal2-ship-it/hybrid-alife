// Immersed-boundary coupling (Peskin 2002, Acta Numerica 11:479) to the D2Q9 BGK solver.
// The fluid step is lbm.cpp's own step_once (included, not copied), so the distribution layout
// is identical to the live solver's: f[(y*nx+x)*9+q]. Lattice node (x, y) sits at position
// (x, y). Lagrangian points are unwrapped; only grid access is periodic.
//
// Structure forces are gradients of an energy made of edge lengths and signed joint angles.
// Both are invariant under rigid translation and rotation, so internal forces have zero net
// force and zero net torque whatever the (time dependent) actuation does: self-propulsion
// comes only from the fluid. Tethers are the one external force (for drag benchmarks).
//
// One step: forces at X^n, t^n -> spread (4-point delta) -> LBM step with Guo forcing ->
// interpolate u = (sum f c + F/2) / rho at X^n -> X^{n+1} = X^n + u.
#include "contact.cpp"
#include "lbm.cpp"

#include <cmath>
#include <limits>

extern "C" {

struct IbStructure {
  int32_t n_points;
  int32_t n_edges;
  const int32_t* edges;      // n_edges*2 (i, j)
  const double* edge_rest;   // rest length
  const double* edge_k;      // stiffness
  const double* edge_shift;  // n_edges*2, added to X[j] - X[i] (closes periodic filaments)
  int32_t n_angles;
  const int32_t* angles;     // n_angles*3 (i, j, k), joint at j
  // rest angle(t) = base + ramp(t) * amp * sin(omega * t - phase)
  const double* angle_base;
  const double* angle_amp;
  const double* angle_phase;
  const double* angle_omega;
  const double* angle_k;
  const double* angle_shift;  // n_angles*4: added to X[i]-X[j] (2) and X[k]-X[j] (2)
  int32_t n_tethers;
  const int32_t* tethers;      // point index
  const double* tether_target; // n_tethers*2
  double tether_k;
  const int32_t* body;  // n_points body ids, used by contact
  double contact_radius;  // 0 disables contact
  double contact_k;
  double ramp_steps;  // actuation amplitude rises linearly over this many steps
  int32_t n_planes;
  const double* planes;  // n_planes*3: unit normal into fluid (nx, ny), c
  int32_t n_discs;
  const double* discs;  // n_discs*3: (cx, cy, R)
  double wall_range;
  double wall_k;
};

struct IbReport {
  double max_net_force;   // max over steps of |sum of internal + contact forces|
  double max_net_torque;  // same for torque about the first point (zero for periodic edges)
  double max_force_scale; // max over steps of sum |F_point|, to judge the two above
  double tether_fx, tether_fy;  // total tether force on the structure at the last step
  double min_contact_distance;  // over all steps; contact_radius if never in contact
  double max_speed;             // largest fluid speed seen at the last step
  double min_wall_gap;          // true minimum over all steps and surfaces; +inf if none
};

}  // extern "C"

namespace {

// Peskin's 4-point regularized delta, 1-D factor.
inline double phi4(double r) {
  r = std::fabs(r);
  if (r < 1.0) return 0.125 * (3.0 - 2.0 * r + std::sqrt(1.0 + 4.0 * r - 4.0 * r * r));
  if (r < 2.0)
    return 0.125 * (5.0 - 2.0 * r - std::sqrt(std::fmax(0.0, -7.0 + 12.0 * r - 4.0 * r * r)));
  return 0.0;
}

struct Stencil {
  int node[16];
  double w[16];
};

inline void make_stencil(double x, double y, int nx, int ny, Stencil& s) {
  const int x0 = static_cast<int>(std::floor(x)) - 1, y0 = static_cast<int>(std::floor(y)) - 1;
  for (int b = 0; b < 4; ++b) {
    const int gy = ((y0 + b) % ny + ny) % ny;
    const double wy = phi4(y - (y0 + b));
    for (int a = 0; a < 4; ++a) {
      const int gx = ((x0 + a) % nx + nx) % nx;
      s.node[b * 4 + a] = gy * nx + gx;
      s.w[b * 4 + a] = wy * phi4(x - (x0 + a));
    }
  }
}

// Fills F (n*2) with internal forces and returns the elastic energy; NaN if degenerate.
double internal_forces(const IbStructure& s, const double* X, double t, double* F) {
  double energy = 0.0;
  for (int e = 0; e < s.n_edges; ++e) {
    const int i = s.edges[2 * e], j = s.edges[2 * e + 1];
    const double dx = X[2 * j] - X[2 * i] + s.edge_shift[2 * e];
    const double dy = X[2 * j + 1] - X[2 * i + 1] + s.edge_shift[2 * e + 1];
    const double r = std::hypot(dx, dy);
    if (!(r > 1e-12)) return NAN;
    const double d = r - s.edge_rest[e];
    energy += 0.5 * s.edge_k[e] * d * d;
    const double c = s.edge_k[e] * d / r;
    F[2 * i] += c * dx;
    F[2 * i + 1] += c * dy;
    F[2 * j] -= c * dx;
    F[2 * j + 1] -= c * dy;
  }
  const double ramp = s.ramp_steps > 0 ? std::fmin(1.0, t / s.ramp_steps) : 1.0;
  for (int a = 0; a < s.n_angles; ++a) {
    const int i = s.angles[3 * a], j = s.angles[3 * a + 1], k = s.angles[3 * a + 2];
    // u = X[j] - X[i], v = X[k] - X[j]; theta = 0 for a straight joint.
    const double ux = -(X[2 * i] - X[2 * j] + s.angle_shift[4 * a]);
    const double uy = -(X[2 * i + 1] - X[2 * j + 1] + s.angle_shift[4 * a + 1]);
    const double vx = X[2 * k] - X[2 * j] + s.angle_shift[4 * a + 2];
    const double vy = X[2 * k + 1] - X[2 * j + 1] + s.angle_shift[4 * a + 3];
    const double u2 = ux * ux + uy * uy, v2 = vx * vx + vy * vy;
    if (!(u2 > 1e-24) || !(v2 > 1e-24)) return NAN;
    const double theta = std::atan2(ux * vy - uy * vx, ux * vx + uy * vy);
    const double rest =
        s.angle_base[a] + ramp * s.angle_amp[a] * std::sin(s.angle_omega[a] * t - s.angle_phase[a]);
    const double d = std::remainder(theta - rest, 2.0 * M_PI);
    energy += 0.5 * s.angle_k[a] * d * d;
    const double m = s.angle_k[a] * d;
    const double gix = -uy / u2, giy = ux / u2;  // dtheta/dX_i
    const double gkx = -vy / v2, gky = vx / v2;  // dtheta/dX_k
    F[2 * i] -= m * gix;
    F[2 * i + 1] -= m * giy;
    F[2 * k] -= m * gkx;
    F[2 * k + 1] -= m * gky;
    F[2 * j] += m * (gix + gkx);
    F[2 * j + 1] += m * (giy + gky);
  }
  return energy;
}

bool valid(const IbStructure& s) {
  const int n = s.n_points;
  if (n < 0 || s.n_edges < 0 || s.n_angles < 0 || s.n_tethers < 0 || s.n_planes < 0 ||
      s.n_discs < 0) return false;
  for (int e = 0; e < 2 * s.n_edges; ++e)
    if (s.edges[e] < 0 || s.edges[e] >= n) return false;
  for (int a = 0; a < 3 * s.n_angles; ++a)
    if (s.angles[a] < 0 || s.angles[a] >= n) return false;
  for (int t = 0; t < s.n_tethers; ++t)
    if (s.tethers[t] < 0 || s.tethers[t] >= n) return false;
  if ((s.n_planes || s.n_discs) && (!(s.wall_range > 0) || !(s.wall_k > 0))) return false;
  for (int p = 0; p < 3 * s.n_planes; ++p)
    if (!std::isfinite(s.planes[p])) return false;
  for (int p = 0; p < s.n_planes; ++p)
    if (std::fabs(std::hypot(s.planes[3 * p], s.planes[3 * p + 1]) - 1.0) > 1e-8) return false;
  for (int d = 0; d < 3 * s.n_discs; ++d)
    if (!std::isfinite(s.discs[d])) return false;
  for (int d = 0; d < s.n_discs; ++d)
    if (!(s.discs[3 * d + 2] > 0)) return false;
  return true;
}

}  // namespace

extern "C" {

// Internal (elastic + actuation) forces at time t into F (n*2). Returns energy, NaN if invalid.
double ib_forces(const IbStructure* s, const double* X, double t, double* F) {
  if (!valid(*s)) return NAN;
  for (int i = 0; i < 2 * s->n_points; ++i) F[i] = 0.0;
  return internal_forces(*s, X, t, F);
}

// Spread point forces F (n*2) onto grid (ny*nx*2, overwritten).
void ib_spread(int ny, int nx, int n, const double* X, const double* F, double* grid) {
  for (int i = 0; i < 2 * ny * nx; ++i) grid[i] = 0.0;
  Stencil s;
  for (int p = 0; p < n; ++p) {
    make_stencil(X[2 * p], X[2 * p + 1], nx, ny, s);
    for (int q = 0; q < 16; ++q) {
      grid[2 * s.node[q]] += s.w[q] * F[2 * p];
      grid[2 * s.node[q] + 1] += s.w[q] * F[2 * p + 1];
    }
  }
}

// Advance fluid (f_io: ny*nx*9) and points (X_io: n*2) together by `steps` from time step0.
// Writes u, v, rho (ny*nx) and the point velocities U (n*2) of the last step.
// Returns steps run; -1 invalid arguments; -2 a delta stencil touched a solid node;
// -3 unstable (non-finite state, degenerate joint, or fluid speed above 0.3).
// On a negative return f_io and X_io are left unchanged.
int ib_advance(int ny, int nx, const uint8_t* mask, double nu, double bx, double by,
               const IbStructure* sp, int steps, double step0, double* f_io, double* X_io,
               double* U, double* u_out, double* v_out, double* rho_out, IbReport* rep) {
  const IbStructure& s = *sp;
  const double tau = 3.0 * nu + 0.5;
  if (ny < 4 || nx < 4 || !(tau > 0.5) || steps < 0 || !valid(s)) return -1;
  // Keeps cells * 9 well inside int, which the shared fluid step indexes with.
  if (static_cast<long long>(ny) * nx > (1LL << 26)) return -1;
  const int n = s.n_points, cells = ny * nx;
  const bool contact_on = s.contact_radius > 0 && s.contact_k > 0;
  if (contact_on && !(nx > 2 * s.contact_radius && ny > 2 * s.contact_radius)) return -1;
  for (int i = 0; i < 2 * n; ++i)
    if (!std::isfinite(X_io[i]) || std::fabs(X_io[i]) > 1e8) return -1;

  std::vector<double> f(f_io, f_io + cells * Q), g(cells * Q), grid(2 * cells, 0.0);
  std::vector<double> X(X_io, X_io + 2 * n), F(2 * n, 0.0);
  std::vector<Stencil> st(n);
  std::vector<int> touched;
  touched.reserve(16 * static_cast<size_t>(n));
  contact::Grid hash;
  *rep = IbReport{0, 0, 0, 0, 0, contact_on ? s.contact_radius : 0.0, 0,
                  std::numeric_limits<double>::infinity()};

  for (int step = 0; step < steps; ++step) {
    F.assign(2 * n, 0.0);
    if (std::isnan(internal_forces(s, X.data(), step0 + step, F.data()))) return -3;
    if (contact_on) {
      double dmin = s.contact_radius;
      contact::add_forces(n, X.data(), s.body, nx, ny, s.contact_radius, s.contact_k, false, hash,
                          F.data(), &dmin);
      rep->min_contact_distance = std::fmin(rep->min_contact_distance, dmin);
    }
    double sx = 0, sy = 0, torque = 0, scale = 0;
    for (int p = 0; p < n; ++p) {
      sx += F[2 * p];
      sy += F[2 * p + 1];
      scale += std::hypot(F[2 * p], F[2 * p + 1]);
      // Contact between periodic images makes unwrapped torque meaningless; see docs.
      torque += (X[2 * p] - X[0]) * F[2 * p + 1] - (X[2 * p + 1] - X[1]) * F[2 * p];
    }
    rep->max_net_force = std::fmax(rep->max_net_force, std::hypot(sx, sy));
    rep->max_net_torque = std::fmax(rep->max_net_torque, std::fabs(torque));
    rep->max_force_scale = std::fmax(rep->max_force_scale, scale);
    rep->tether_fx = rep->tether_fy = 0.0;
    for (int t = 0; t < s.n_tethers; ++t) {
      const int p = s.tethers[t];
      const double fx = -s.tether_k * (X[2 * p] - s.tether_target[2 * t]);
      const double fy = -s.tether_k * (X[2 * p + 1] - s.tether_target[2 * t + 1]);
      F[2 * p] += fx;
      F[2 * p + 1] += fy;
      rep->tether_fx += fx;
      rep->tether_fy += fy;
    }
    for (int p = 0; p < n; ++p) {
      for (int w = 0; w < s.n_planes; ++w) {
        const double* plane = s.planes + 3 * w;
        const double gap = plane[0] * X[2 * p] + plane[1] * X[2 * p + 1] - plane[2];
        rep->min_wall_gap = std::fmin(rep->min_wall_gap, gap);
        if (gap < s.wall_range) {
          const double magnitude = s.wall_k * (s.wall_range - gap);
          F[2 * p] += magnitude * plane[0];
          F[2 * p + 1] += magnitude * plane[1];
        }
      }
      for (int w = 0; w < s.n_discs; ++w) {
        const double* disc = s.discs + 3 * w;
        double dx = X[2 * p] - disc[0], dy = X[2 * p + 1] - disc[1];
        dx -= nx * std::nearbyint(dx / nx);
        dy -= ny * std::nearbyint(dy / ny);
        const double distance = std::hypot(dx, dy);
        const double gap = distance - disc[2];
        rep->min_wall_gap = std::fmin(rep->min_wall_gap, gap);
        if (gap < s.wall_range && distance > 0) {
          const double magnitude = s.wall_k * (s.wall_range - gap) / distance;
          F[2 * p] += magnitude * dx;
          F[2 * p + 1] += magnitude * dy;
        }
      }
    }

    for (int node : touched) grid[2 * node] = grid[2 * node + 1] = 0.0;
    touched.clear();
    for (int p = 0; p < n; ++p) {
      if (!std::isfinite(X[2 * p]) || !std::isfinite(X[2 * p + 1])) return -3;
      make_stencil(X[2 * p], X[2 * p + 1], nx, ny, st[p]);
      for (int q = 0; q < 16; ++q) {
        const int node = st[p].node[q];
        if (mask[node] && st[p].w[q] > 0) return -2;
        grid[2 * node] += st[p].w[q] * F[2 * p];
        grid[2 * node + 1] += st[p].w[q] * F[2 * p + 1];
        touched.push_back(node);
      }
    }

    step_once(ny, nx, mask, 1.0 / tau, bx, by, grid.data(), f, g);

    // Only the stencil nodes need a velocity.
    for (int p = 0; p < n; ++p) {
      double ux = 0, uy = 0;
      for (int q = 0; q < 16; ++q) {
        if (st[p].w[q] == 0) continue;
        const int node = st[p].node[q];
        double rho = 0, mx = 0, my = 0;
        for (int d = 0; d < Q; ++d) {
          const double v = f[node * Q + d];
          rho += v;
          mx += v * CX[d];
          my += v * CY[d];
        }
        mx += 0.5 * (rho * bx + grid[2 * node]);
        my += 0.5 * (rho * by + grid[2 * node + 1]);
        ux += st[p].w[q] * mx / rho;
        uy += st[p].w[q] * my / rho;
      }
      if (!(std::hypot(ux, uy) <= 0.3)) return -3;
      U[2 * p] = ux;
      U[2 * p + 1] = uy;
    }
    for (int i = 0; i < 2 * n; ++i) X[i] += U[i];
  }

  macroscopic(ny, nx, mask, bx, by, grid.data(), f, u_out, v_out, rho_out);
  for (int i = 0; i < cells; ++i) {
    const double speed = std::hypot(u_out[i], v_out[i]);
    if (!(speed <= 0.3)) return -3;
    rep->max_speed = std::fmax(rep->max_speed, speed);
  }
  for (int i = 0; i < cells * Q; ++i) f_io[i] = f[i];
  for (int i = 0; i < 2 * n; ++i) X_io[i] = X[i];
  return steps;
}

}  // extern "C"
