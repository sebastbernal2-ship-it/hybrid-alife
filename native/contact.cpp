// Soft-sphere contact between Lagrangian points of different bodies, periodic domain,
// found with a spatial hash (uniform cell list, cell edge >= cutoff). Each pair gets equal
// and opposite forces along the line of centres, so the kernel adds zero net force and zero
// net torque. It is a penalty method: overlap is limited by stiffness, not forbidden.
// C ABI for ctypes. Also included by ib.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace contact {

inline double min_image(double d, double L) { return d - L * std::floor(d / L + 0.5); }

struct Grid {
  int cx = 0, cy = 0;
  std::vector<int> head, next, cell;
};

// Adds contact forces to F (n*2). Returns the number of contacting pairs; *min_dist gets the
// smallest inter-body distance among candidate pairs closer than rc (rc if none).
inline long add_forces(int n, const double* X, const int32_t* body, double Lx, double Ly,
                       double rc, double k, bool brute, Grid& g, double* F, double* min_dist) {
  long pairs = 0;
  double dmin = rc;
  auto visit = [&](int i, int j) {
    if (body[i] == body[j]) return;
    const double dx = min_image(X[2 * i] - X[2 * j], Lx);
    const double dy = min_image(X[2 * i + 1] - X[2 * j + 1], Ly);
    const double r2 = dx * dx + dy * dy;
    if (r2 >= rc * rc) return;
    const double r = std::sqrt(r2);
    ++pairs;
    dmin = std::min(dmin, r);
    if (r == 0.0) return;  // coincident points have no defined normal
    const double s = k * (rc - r) / r;
    F[2 * i] += s * dx;
    F[2 * i + 1] += s * dy;
    F[2 * j] -= s * dx;
    F[2 * j + 1] -= s * dy;
  };
  const int cx = static_cast<int>(std::min(Lx / rc, 4096.0));
  const int cy = static_cast<int>(std::min(Ly / rc, 4096.0));
  // With fewer than 3 cells per axis the 3x3 stencil would visit a neighbour twice.
  if (brute || cx < 3 || cy < 3) {
    for (int i = 0; i < n; ++i)
      for (int j = i + 1; j < n; ++j) visit(i, j);
  } else {
    g.cx = cx;
    g.cy = cy;
    g.head.assign(static_cast<size_t>(cx) * cy, -1);
    g.next.assign(n, -1);
    g.cell.assign(n, 0);
    auto index = [](double x, double L, int c) {
      const double w = x - L * std::floor(x / L);
      return std::min(c - 1, std::max(0, static_cast<int>(w / L * c)));
    };
    for (int i = 0; i < n; ++i) {
      const int c = index(X[2 * i + 1], Ly, cy) * cx + index(X[2 * i], Lx, cx);
      g.cell[i] = c;
      g.next[i] = g.head[c];
      g.head[c] = i;
    }
    for (int i = 0; i < n; ++i) {
      const int ix = g.cell[i] % cx, iy = g.cell[i] / cx;
      for (int oy = -1; oy <= 1; ++oy)
        for (int ox = -1; ox <= 1; ++ox) {
          const int c = ((iy + oy + cy) % cy) * cx + (ix + ox + cx) % cx;
          for (int j = g.head[c]; j >= 0; j = g.next[j])
            if (j > i) visit(i, j);
        }
    }
  }
  if (min_dist) *min_dist = dmin;
  return pairs;
}

}  // namespace contact

extern "C" {

// F (n*2) is overwritten. Returns the pair count, or -1 on invalid arguments.
long contact_forces(int n, const double* X, const int32_t* body, double Lx, double Ly, double rc,
                    double k, int brute, double* F, double* min_dist) {
  if (n < 0 || !(rc > 0) || !(Lx > 2 * rc) || !(Ly > 2 * rc) || !(k >= 0)) return -1;
  for (int i = 0; i < 2 * n; ++i) {
    if (!std::isfinite(X[i])) return -1;
    F[i] = 0.0;
  }
  contact::Grid g;
  return contact::add_forces(n, X, body, Lx, Ly, rc, k, brute != 0, g, F, min_dist);
}

}  // extern "C"
