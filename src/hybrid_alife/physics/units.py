"""Lattice-to-SI mapping for the D2Q9 fluid, with the dimensionless numbers that matter.

Choose the cell size `dx_m` and lattice viscosity `nu_lu`; matching the fluid's kinematic
viscosity then fixes the time step, dt = nu_lu * dx^2 / nu_si. The solver is 2-D, so
forces are per metre of out-of-plane depth. See docs/extensions/immersed_boundary.md.
"""

from __future__ import annotations

from dataclasses import dataclass

WATER_NU = 1.0e-6  # m^2/s, water near 20 C
WATER_RHO = 998.0  # kg/m^3


@dataclass(frozen=True)
class UnitMap:
    dx_m: float
    nu_lu: float = 1.0 / 6.0
    nu_si: float = WATER_NU
    rho_si: float = WATER_RHO  # one lattice density unit

    def __post_init__(self) -> None:
        if min(self.dx_m, self.nu_lu, self.nu_si, self.rho_si) <= 0:
            raise ValueError("unit map parameters must all be positive")

    @property
    def dt_s(self) -> float:
        return self.nu_lu * self.dx_m**2 / self.nu_si

    def velocity_to_lu(self, m_per_s: float) -> float:
        return m_per_s * self.dt_s / self.dx_m

    def velocity_to_si(self, lu: float) -> float:
        return lu * self.dx_m / self.dt_s

    def steps_per_period(self, hz: float) -> float:
        return 1.0 / (hz * self.dt_s)

    def force_per_depth_to_si(self, lu: float) -> float:
        """Lattice force (per unit depth) to N/m."""
        return lu * self.rho_si * self.dx_m**3 / self.dt_s**2

    def reynolds(self, speed_si: float, length_si: float) -> float:
        return speed_si * length_si / self.nu_si

    @staticmethod
    def peclet(speed_si: float, length_si: float, diffusivity_si: float) -> float:
        return speed_si * length_si / diffusivity_si


def oscillatory_reynolds(omega: float, k: float, nu: float) -> float:
    """omega / (nu k^2): unsteady inertia against viscosity for a wave (any unit system)."""
    return omega / (nu * k**2)
