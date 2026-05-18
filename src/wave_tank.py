"""
wave_tank.py — Numerical Wave Tank: wave generation module.

Provides a standalone vectorized dispersion solver, a first-order (Airy)
regular wave, and a JONSWAP-based irregular sea state.

Key design choices
  1. Standalone vectorized dispersion solver — solves all frequencies in one call.
  2. RegularWave — wave-regime classification, group velocity, energy/power,
     particle kinematics (u, w), accelerations, dynamic pressure, breaking checks.
  3. IrregularWave_JONSWAP — local RNG (no global seed side-effects), vectorized
     dispersion, components stored as flat numpy arrays, fully vectorized
     surface_elevation and particle_velocity via numpy broadcasting.
"""

import warnings
import numpy as np


# ---------------------------------------------------------------------------
# Standalone vectorized dispersion solver
# ---------------------------------------------------------------------------

def solve_dispersion(omega, d, g=9.81, tol=1e-10, max_iter=50):
    """
    Solve omega^2 = g * k * tanh(k * d) for k.

    Accepts scalar or array omega. Returns k with the same shape.
    Uses Hunt (1979) as an initial guess, then Newton-Raphson.
    """
    omega = np.asarray(omega, dtype=float)
    scalar = omega.ndim == 0
    omega = np.atleast_1d(omega)

    y = omega**2 * d / g
    kd0 = np.sqrt(y**2 + y / (1.0 + 0.6522*y + 0.4622*y**2 + 0.0864*y**4 + 0.0675*y**5))
    k = kd0 / d

    for _ in range(max_iter):
        kd = k * d
        tanh_kd = np.tanh(kd)
        f = g * k * tanh_kd - omega**2
        fp = g * (tanh_kd + kd * (1.0 - tanh_kd**2))
        dk = f / fp
        k -= dk
        if np.all(np.abs(dk) < tol * np.abs(k)):
            break
    else:
        warnings.warn("solve_dispersion: Newton-Raphson did not converge for all frequencies.")

    return float(k[0]) if scalar else k


# ---------------------------------------------------------------------------
# RegularWave
# ---------------------------------------------------------------------------

class RegularWave:
    """First-Order (Airy) Regular Wave with full kinematic output."""

    G: float = 9.81    # gravitational acceleration [m/s^2]
    RHO: float = 1025.0  # seawater density [kg/m^3]

    def __init__(self, H: float, T: float, d: float, phase: float = 0.0):
        if H <= 0 or T <= 0 or d <= 0:
            raise ValueError("H, T, and d must be positive.")

        self.H = H
        self.T = T
        self.d = d
        self.phase = phase

        self.omega = 2.0 * np.pi / T
        self.k = solve_dispersion(self.omega, d, self.G)
        self.L = 2.0 * np.pi / self.k
        self.C = self.omega / self.k

        n = 0.5 * (1.0 + 2.0 * self.k * d / np.sinh(2.0 * self.k * d))
        self.Cg = n * self.C  # group velocity [m/s]

        kd = self.k * d
        if kd < np.pi / 10.0:
            self.regime = "shallow"
        elif kd > np.pi:
            self.regime = "deep"
        else:
            self.regime = "intermediate"

        if H / self.L > 1.0 / 7.0:
            warnings.warn(f"H/L = {H/self.L:.4f} exceeds 1/7 (wave breaking steepness limit).")
        if H / d > 0.78:
            warnings.warn(f"H/d = {H/d:.4f} exceeds 0.78 (depth-limited breaking limit).")

    def __repr__(self) -> str:
        return (f"RegularWave(H={self.H}m, T={self.T}s, d={self.d}m, "
                f"L={self.L:.2f}m, C={self.C:.2f}m/s, Cg={self.Cg:.2f}m/s, "
                f"regime='{self.regime}')")

    @property
    def a(self) -> float:
        """Amplitude [m]."""
        return self.H / 2.0

    @property
    def energy_density(self) -> float:
        """Mean wave energy density E = (1/8)*rho*g*H^2 [J/m^2]."""
        return 0.125 * self.RHO * self.G * self.H**2

    @property
    def power_flux(self) -> float:
        """Wave power (energy flux) P = E * Cg [W/m]."""
        return self.energy_density * self.Cg

    # ------------------------------------------------------------------
    # Surface elevation
    # ------------------------------------------------------------------

    def surface_elevation(self, x, t):
        """Free-surface elevation eta(x, t) [m]."""
        return self.a * np.cos(self.k * x - self.omega * t + self.phase)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _theta(self, x, t):
        return self.k * x - self.omega * t + self.phase

    def _cosh_ratio(self, z):
        """cosh(k*(z+d)) / sinh(k*d)  —  used for horizontal velocity."""
        return np.cosh(self.k * (z + self.d)) / np.sinh(self.k * self.d)

    def _sinh_ratio(self, z):
        """sinh(k*(z+d)) / sinh(k*d)  —  used for vertical velocity."""
        return np.sinh(self.k * (z + self.d)) / np.sinh(self.k * self.d)

    # ------------------------------------------------------------------
    # Kinematics (Airy wave theory)
    # ------------------------------------------------------------------

    def particle_velocity(self, x, z, t):
        """
        Orbital velocity components (u, w) [m/s].

        z is measured upward from the still-water level; valid for -d <= z <= 0.
        x and t may be scalars or matching numpy arrays.
        """
        theta = self._theta(x, t)
        u = self.a * self.omega * self._cosh_ratio(z) * np.cos(theta)
        w = self.a * self.omega * self._sinh_ratio(z) * np.sin(theta)
        return u, w

    def particle_acceleration(self, x, z, t):
        """
        Orbital acceleration components (ax, az) [m/s^2].
        """
        theta = self._theta(x, t)
        ax =  self.a * self.omega**2 * self._cosh_ratio(z) * np.sin(theta)
        az = -self.a * self.omega**2 * self._sinh_ratio(z) * np.cos(theta)
        return ax, az

    def dynamic_pressure(self, x, z, t):
        """
        Dynamic (wave-induced) pressure p_dyn [Pa].
        p = rho * g * a * cosh(k*(z+d)) / cosh(k*d) * cos(theta)
        """
        p_factor = np.cosh(self.k * (z + self.d)) / np.cosh(self.k * self.d)
        return self.RHO * self.G * self.a * p_factor * np.cos(self._theta(x, t))


# ---------------------------------------------------------------------------
# IrregularWave_JONSWAP
# ---------------------------------------------------------------------------

class IrregularWave_JONSWAP:
    """
    Irregular sea state from a JONSWAP spectrum.

    All wave components are stored as plain numpy arrays instead of individual
    RegularWave objects. surface_elevation and particle_velocity are fully
    vectorized — no Python-level loop over components.
    """

    G: float = 9.81
    RHO: float = 1025.0

    def __init__(
        self,
        Hs: float,
        Tp: float,
        d: float,
        gamma: float = 3.3,
        T_repeat: float = 10800.0,
        f_min_factor: float = 0.5,
        f_max_factor: float = 5.0,
        seed=None,
    ):
        if Hs <= 0 or Tp <= 0 or d <= 0:
            raise ValueError("Hs, Tp, and d must be positive.")

        self.Hs = Hs
        self.Tp = Tp
        self.d = d
        self.gamma = gamma

        fp = 1.0 / Tp
        df = 1.0 / T_repeat
        self.f = np.arange(f_min_factor * fp, f_max_factor * fp, df)
        self.omega = 2.0 * np.pi * self.f

        # Solve dispersion for ALL components in a single vectorized call
        self.k = solve_dispersion(self.omega, d, self.G)

        S_f = self._jonswap()

        # Local RNG — does not mutate the global numpy random state
        rng = np.random.default_rng(seed)
        scale = np.sqrt(S_f * df)
        self.amplitudes = rng.rayleigh(scale=scale)
        self.phases = rng.uniform(-np.pi, np.pi, len(self.f))

    def __repr__(self) -> str:
        return (f"IrregularWave_JONSWAP(Hs={self.Hs}m, Tp={self.Tp}s, d={self.d}m, "
                f"gamma={self.gamma}, N={len(self.f)} components, "
                f"actual_hs={self.actual_hs:.3f}m)")

    # ------------------------------------------------------------------
    # Spectrum
    # ------------------------------------------------------------------

    def _jonswap(self) -> np.ndarray:
        fp = 1.0 / self.Tp
        sigma = np.where(self.f <= fp, 0.07, 0.09)
        S_pm = ((5.0 / 16.0) * self.Hs**2 * fp**4 * self.f**-5
                * np.exp(-1.25 * (fp / self.f)**4))
        J = self.gamma ** np.exp(-0.5 * ((self.f - fp) / (sigma * fp))**2)
        A_gamma = 1.0 - 0.287 * np.log(self.gamma)
        return A_gamma * S_pm * J

    @property
    def actual_hs(self) -> float:
        """Significant wave height from the generated amplitudes: Hs = 4*sqrt(m0)."""
        m0 = 0.5 * np.sum(self.amplitudes**2)
        return 4.0 * np.sqrt(m0)

    # ------------------------------------------------------------------
    # Surface elevation (vectorized)
    # ------------------------------------------------------------------

    def surface_elevation(self, x, t):
        """
        Free-surface elevation eta(x, t) [m] via fully vectorized numpy ops.

        Accepted call signatures:
          surface_elevation(x_array, t_scalar)  -> shape (M,)
          surface_elevation(x_scalar, t_array)  -> shape (P,)
          surface_elevation(x_scalar, t_scalar) -> float
        """
        x = np.asarray(x, dtype=float)
        t = np.asarray(t, dtype=float)

        # Both scalar
        if x.ndim == 0 and t.ndim == 0:
            phase = self.k * float(x) - self.omega * float(t) + self.phases
            return float(np.dot(self.amplitudes, np.cos(phase)))

        # Spatial sweep at a fixed time: x is (M,), t is scalar
        if t.ndim == 0:
            # phase[n, m] = k[n]*x[m] - omega[n]*t + phases[n]
            phase = (self.k[:, np.newaxis] * x[np.newaxis, :] #type: ignore
                     - (self.omega * float(t))[:, np.newaxis]
                     + self.phases[:, np.newaxis])
            return np.einsum('n,nm->m', self.amplitudes, np.cos(phase))

        # Time series at a fixed location: x is scalar, t is (P,)
        if x.ndim == 0:
            # phase[n, p] = k[n]*x - omega[n]*t[p] + phases[n]
            phase = (self.k[:, np.newaxis] * float(x) #type: ignore
                     - self.omega[:, np.newaxis] * t[np.newaxis, :]
                     + self.phases[:, np.newaxis])
            return np.einsum('n,np->p', self.amplitudes, np.cos(phase))

        raise ValueError("x and t cannot both be arrays; pass one array and one scalar.")

    # ------------------------------------------------------------------
    # Kinematics (vectorized)
    # ------------------------------------------------------------------

    def particle_velocity(self, x: float, z: float, t: float):
        """
        Orbital velocity (u, w) [m/s] at a point (x, z, t).

        All three arguments must be scalars.
        z is measured upward from the still-water level; valid for -d <= z <= 0.
        """
        cosh_f = np.cosh(self.k * (z + self.d)) / np.sinh(self.k * self.d)
        sinh_f = np.sinh(self.k * (z + self.d)) / np.sinh(self.k * self.d)
        phase = self.k * x - self.omega * t + self.phases
        u = np.dot(self.amplitudes * self.omega * cosh_f, np.cos(phase))
        w = np.dot(self.amplitudes * self.omega * sinh_f, np.sin(phase))
        return float(u), float(w)
