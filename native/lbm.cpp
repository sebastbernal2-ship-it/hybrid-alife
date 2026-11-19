// D2Q9 BGK lattice-Boltzmann solver with a Guo body force.
// Periodic in x and y; solid cells (mask != 0) use halfway bounce-back, so
// channel walls are expressed by masking rows. C ABI for ctypes.
//
// Two entry points share one step routine:
//   lbm_solve   : one-shot run from rest to (near) steady state.
//   lbm_init / lbm_advance : stateful; the distributions live in a caller-owned
//                 buffer, so state is plain numpy (picklable, no native handle).
// Forces: a uniform body acceleration (bx, by) times local density, plus an
// optional per-cell force density field (fx, fy per cell, same units as
// density * acceleration). Per step the raw momentum sum_i f_i c_i of a
// periodic fluid changes by exactly the total applied force.
#include <cstdint>
#include <vector>

namespace {
constexpr int Q = 9;
constexpr int CX[Q] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
constexpr int CY[Q] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
constexpr int OPP[Q] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
constexpr double W[Q] = {4.0 / 9,  1.0 / 9,  1.0 / 9,  1.0 / 9, 1.0 / 9,
                         1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36};

inline double feq(int q, double rho, double ux, double uy) {
  const double cu = CX[q] * ux + CY[q] * uy;
  return W[q] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * (ux * ux + uy * uy));
}

// One collide+stream step on f (scratch g), in place via swap in the caller.
void step_once(int h, int w, const uint8_t* mask, double omega, double bx, double by,
               const double* force, std::vector<double>& f, std::vector<double>& g) {
  const double pref = 1.0 - 0.5 * omega;
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      double* fi = &f[i * Q];
      if (mask[i]) continue;
      double rho = 0, mx = 0, my = 0;
      for (int q = 0; q < Q; ++q) {
        rho += fi[q];
        mx += fi[q] * CX[q];
        my += fi[q] * CY[q];
      }
      double Fx = rho * bx, Fy = rho * by;
      if (force) { Fx += force[2 * i]; Fy += force[2 * i + 1]; }
      const double ux = (mx + 0.5 * Fx) / rho;
      const double uy = (my + 0.5 * Fy) / rho;
      for (int q = 0; q < Q; ++q) {
        const double cu = CX[q] * ux + CY[q] * uy;
        const double ft = pref * W[q] *
            (3.0 * ((CX[q] - ux) * Fx + (CY[q] - uy) * Fy) + 9.0 * cu * (CX[q] * Fx + CY[q] * Fy));
        fi[q] += -omega * (fi[q] - feq(q, rho, ux, uy)) + ft;
      }
    }
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const int i = y * w + x;
      if (mask[i]) continue;
      for (int q = 0; q < Q; ++q) {
        const int xn = (x + CX[q] + w) % w, yn = (y + CY[q] + h) % h;
        const int j = yn * w + xn;
        if (mask[j]) g[i * Q + OPP[q]] = f[i * Q + q];
        else g[j * Q + q] = f[i * Q + q];
      }
    }
  f.swap(g);
}

void macroscopic(int h, int w, const uint8_t* mask, double bx, double by, const double* force,
                 const std::vector<double>& f, double* u, double* v, double* rho_out) {
  for (int i = 0; i < h * w; ++i) {
    if (mask[i]) { u[i] = v[i] = 0; rho_out[i] = 1.0; continue; }
    double rho = 0, mx = 0, my = 0;
    for (int q = 0; q < Q; ++q) {
      rho += f[i * Q + q];
      mx += f[i * Q + q] * CX[q];
      my += f[i * Q + q] * CY[q];
    }
    double Fx = rho * bx, Fy = rho * by;
    if (force) { Fx += force[2 * i]; Fy += force[2 * i + 1]; }
    rho_out[i] = rho;
    u[i] = (mx + 0.5 * Fx) / rho;
    v[i] = (my + 0.5 * Fy) / rho;
  }
}
}  // namespace

extern "C" {

// mask: h*w uint8 (1 = solid). Outputs u, v, rho: h*w doubles (row-major, y*w+x).
// nu: lattice kinematic viscosity; fx, fy: body force per unit density.
// Returns the number of steps run, or -1 on invalid arguments (tau <= 0.5).
int lbm_solve(int h, int w, const uint8_t* mask, double nu, double fx, double fy,
              int steps, double* u, double* v, double* rho_out) {
  const double tau = 3.0 * nu + 0.5;
  if (h < 3 || w < 3 || tau <= 0.5 || steps < 0) return -1;
  const int n = h * w;
  std::vector<double> f(n * Q), g(n * Q);
  for (int i = 0; i < n; ++i)
    for (int q = 0; q < Q; ++q) f[i * Q + q] = feq(q, 1.0, 0.0, 0.0);
  for (int s = 0; s < steps; ++s) step_once(h, w, mask, 1.0 / tau, fx, fy, nullptr, f, g);
  macroscopic(h, w, mask, fx, fy, nullptr, f, u, v, rho_out);
  return steps;
}

// Fill f (h*w*9) with the equilibrium of density rho0 and velocity (u, v) per cell.
void lbm_init(int h, int w, double rho0, const double* u, const double* v, double* f) {
  for (int i = 0; i < h * w; ++i)
    for (int q = 0; q < Q; ++q) f[i * Q + q] = feq(q, rho0, u[i], v[i]);
}

// Advance the caller-owned distributions f by `steps`, applying the same force every step.
// force: nullptr or h*w*2 doubles (force density, x then y per cell). Writes u, v, rho.
// Returns steps run, or -1 on invalid arguments.
int lbm_advance(int h, int w, const uint8_t* mask, double nu, double bx, double by,
                const double* force, int steps, double* f_io, double* u, double* v,
                double* rho_out) {
  const double tau = 3.0 * nu + 0.5;
  if (h < 3 || w < 3 || tau <= 0.5 || steps < 0) return -1;
  const int n = h * w;
  std::vector<double> f(f_io, f_io + n * Q), g(n * Q);
  for (int s = 0; s < steps; ++s) step_once(h, w, mask, 1.0 / tau, bx, by, force, f, g);
  for (int i = 0; i < n * Q; ++i) f_io[i] = f[i];
  macroscopic(h, w, mask, bx, by, force, f, u, v, rho_out);
  return steps;
}

}  // extern "C"
