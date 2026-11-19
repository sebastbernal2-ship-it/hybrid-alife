// D2Q9 BGK lattice-Boltzmann solver with a Guo body force.
// Periodic in x and y; solid cells (mask != 0) use halfway bounce-back, so
// channel walls are expressed by masking rows. C ABI for ctypes.
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

  const double omega = 1.0 / tau;
  const double pref = 1.0 - 0.5 * omega;
  for (int s = 0; s < steps; ++s) {
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
        const double ux = (mx + 0.5 * rho * fx) / rho;
        const double uy = (my + 0.5 * rho * fy) / rho;
        for (int q = 0; q < Q; ++q) {
          const double cu = CX[q] * ux + CY[q] * uy;
          const double force = pref * W[q] *
              (3.0 * ((CX[q] - ux) * fx + (CY[q] - uy) * fy) + 9.0 * cu * (CX[q] * fx + CY[q] * fy)) * rho;
          fi[q] += -omega * (fi[q] - feq(q, rho, ux, uy)) + force;
        }
      }
    // Streaming with halfway bounce-back at solid neighbours.
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

  for (int i = 0; i < n; ++i) {
    if (mask[i]) { u[i] = v[i] = 0; rho_out[i] = 1.0; continue; }
    double rho = 0, mx = 0, my = 0;
    for (int q = 0; q < Q; ++q) {
      rho += f[i * Q + q];
      mx += f[i * Q + q] * CX[q];
      my += f[i * Q + q] * CY[q];
    }
    rho_out[i] = rho;
    u[i] = (mx + 0.5 * rho * fx) / rho;
    v[i] = (my + 0.5 * rho * fy) / rho;
  }
  return steps;
}

}  // extern "C"
