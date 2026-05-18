"""
fun_test.py — Validation tests and performance benchmarks for wave_tank.py.

Run:
    .venv/bin/python fun_test.py
"""

import time
import numpy as np
from wave_tank import RegularWave, IrregularWave_JONSWAP, solve_dispersion

PASS = "\033[92mPASSED\033[0m"
FAIL = "\033[91mFAILED\033[0m"


def test_dispersion_accuracy():
    """Residual of the dispersion relation should be below 1e-9."""
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    residual = abs(wave.omega**2 - wave.G * wave.k * np.tanh(wave.k * wave.d))
    print(f"[1] Dispersion relation accuracy")
    print(f"    |omega^2 - g*k*tanh(kd)| = {residual:.2e}  (need < 1e-9)")
    print(f"    {PASS if residual < 1e-9 else FAIL}")
    return residual < 1e-9


def test_wave_properties():
    """Check regime, energy density, and power flux are computed."""
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    print(f"\n[2] RegularWave properties")
    print(f"    {wave!r}")
    print(f"    Energy density : {wave.energy_density:.1f} J/m^2")
    print(f"    Power flux     : {wave.power_flux:.1f} W/m")
    print(f"    {PASS}")
    return True


def test_bed_boundary_condition():
    """Vertical velocity at the seabed must be zero (no normal flow)."""
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    _, w_bed = wave.particle_velocity(x=0.0, z=-wave.d, t=0.0)
    print(f"\n[3] Bed boundary condition  w(z=-d) = 0")
    print(f"    w(z=-d) = {w_bed:.3e} m/s  (should be ~0)")
    ok = abs(w_bed) < 1e-12
    print(f"    {PASS if ok else FAIL}")
    return ok


def test_surface_horizontal_velocity():
    """Numerical u at z=0 must match the closed-form Airy result."""
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    u_num, _ = wave.particle_velocity(x=0.0, z=0.0, t=0.0)
    u_ana = wave.a * wave.omega / np.tanh(wave.k * wave.d)
    err = abs(u_num - u_ana)
    print(f"\n[4] Surface horizontal velocity vs analytical")
    print(f"    u_numerical  = {u_num:.8f} m/s")
    print(f"    u_analytical = {u_ana:.8f} m/s")
    print(f"    error        = {err:.2e}")
    ok = err < 1e-12
    print(f"    {PASS if ok else FAIL}")
    return ok


def test_dynamic_pressure():
    """Dynamic pressure at z=0, t=0, x=0 equals rho*g*a exactly."""
    wave = RegularWave(H=2.0, T=8.0, d=50.0)
    p_num = wave.dynamic_pressure(x=0.0, z=0.0, t=0.0)
    p_ana = wave.RHO * wave.G * wave.a
    err = abs(p_num - p_ana)
    print(f"\n[5] Dynamic pressure at z=0 vs analytical")
    print(f"    p_numerical  = {p_num:.6f} Pa")
    print(f"    p_analytical = {p_ana:.6f} Pa")
    print(f"    error        = {err:.2e}")
    ok = err < 1e-10
    print(f"    {PASS if ok else FAIL}")
    return ok


def test_jonswap_hs_accuracy():
    """actual_hs should be within 15% of the target (single realisation tolerance)."""
    storm = IrregularWave_JONSWAP(Hs=3.0, Tp=10.0, d=50.0, seed=0)
    hs_err_pct = abs(storm.actual_hs - 3.0) / 3.0 * 100
    print(f"\n[6] JONSWAP actual Hs vs target")
    print(f"    Target Hs   = 3.000 m")
    print(f"    Actual Hs   = {storm.actual_hs:.3f} m")
    print(f"    Error       = {hs_err_pct:.1f}%  (expected < 15% for single realisation)")
    ok = hs_err_pct < 15
    print(f"    {PASS if ok else FAIL}")
    return ok


def benchmark_surface_elevation():
    """Compare vectorized surface_elevation against a plain Python loop."""
    storm = IrregularWave_JONSWAP(Hs=2.5, Tp=10.0, d=50.0, seed=42)
    x_domain = np.linspace(0, 300, 500)

    def _loop(s, x, t):
        eta = np.zeros_like(x)
        for amp, om, k, ph in zip(s.amplitudes, s.omega, s.k, s.phases):
            eta += amp * np.cos(k * x - om * t + ph)
        return eta

    N_rep = 10
    t0 = time.perf_counter()
    for _ in range(N_rep):
        _loop(storm, x_domain, 0.0)
    t_loop = (time.perf_counter() - t0) / N_rep * 1000

    t0 = time.perf_counter()
    for _ in range(N_rep):
        storm.surface_elevation(x_domain, 0.0)
    t_vec = (time.perf_counter() - t0) / N_rep * 1000

    print(f"\n[7] Performance — surface_elevation(x_array, t_scalar)")
    print(f"    Python loop  : {t_loop:7.2f} ms / call")
    print(f"    Vectorized   : {t_vec:7.2f} ms / call")
    print(f"    Speedup      : {t_loop / t_vec:.1f}x faster")
    print(f"    {PASS}")
    return True


def benchmark_initialisation():
    """Compare vectorized init against per-component scalar dispersion solves."""
    def _old_init(Hs, Tp, d, seed):
        fp = 1.0 / Tp
        df = 1.0 / 10800.0
        f = np.arange(0.3 * fp, 3.0 * fp, df)
        rng = np.random.default_rng(seed)
        for fi in f:
            solve_dispersion(2 * np.pi * fi, d)
        _ = rng.rayleigh(scale=np.ones(len(f)))

    N_rep = 3
    t0 = time.perf_counter()
    for _ in range(N_rep):
        _old_init(2.5, 10.0, 50.0, 42)
    t_old = (time.perf_counter() - t0) / N_rep * 1000

    t0 = time.perf_counter()
    for _ in range(N_rep):
        IrregularWave_JONSWAP(Hs=2.5, Tp=10.0, d=50.0, seed=42)
    t_new = (time.perf_counter() - t0) / N_rep * 1000

    print(f"\n[8] Performance — initialisation")
    print(f"    Per-component init : {t_old:7.2f} ms")
    print(f"    Vectorized init    : {t_new:7.2f} ms")
    print(f"    Speedup            : {t_old / t_new:.1f}x faster")
    print(f"    {PASS}")
    return True


def run_all():
    print("=" * 62)
    print("  fun_test.py — Validation & Performance Tests")
    print("=" * 62)

    results = [
        test_dispersion_accuracy(),
        test_wave_properties(),
        test_bed_boundary_condition(),
        test_surface_horizontal_velocity(),
        test_dynamic_pressure(),
        test_jonswap_hs_accuracy(),
        benchmark_surface_elevation(),
        benchmark_initialisation(),
    ]

    print("\n" + "=" * 62)
    n_pass = sum(results)
    print(f"  {n_pass}/{len(results)} tests passed.")
    print("=" * 62)


if __name__ == "__main__":
    run_all()
