"""
visualise.py — Visualisation utilities for RegularWave and IrregularWave_JONSWAP objects.

Functions
---------
animate_surface(wave, x, t, *, fps=20, save_path=None)
    Animated η(x, t) surface over the given time array.
    Pass x and t as 1-D numpy arrays (e.g. from np.linspace).
    Saves a .gif when save_path is given. Returns the FuncAnimation.

snapshot_time(wave, x, t, *, ax=None)
    Spatial snapshot η(x) at a fixed time t. Returns (fig, ax).

snapshot_space(wave, t, x, *, ax=None)
    Time series η(t) at a fixed position x. Returns (fig, ax).

plot_spectrum(wave, *, ax=None)
    Energy density spectrum S(f).
    IrregularWave_JONSWAP: JONSWAP target curve + empirical spectrum from amplitudes.
    RegularWave: single vertical line at the wave frequency.
    Returns (fig, ax).

Example usage
-------------
    import numpy as np
    from wave_tank import RegularWave, IrregularWave_JONSWAP
    from visualise import animate_surface, snapshot_time, snapshot_space, plot_spectrum

    storm = IrregularWave_JONSWAP(Hs=3.0, Tp=10.0, d=50.0, seed=42)
    x = np.linspace(0, 600, 600)
    t = np.linspace(0, 60, 120)

    ani = animate_surface(storm, x, t, fps=20, save_path="../output/storm.gif")
    fig, ax = snapshot_time(storm, x, t=30.0)
    fig, ax = snapshot_space(storm, t, x=0.0)
    fig, ax = plot_spectrum(storm)
    import matplotlib.pyplot as plt; plt.show()
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ---------------------------------------------------------------------------
# Animation
# ---------------------------------------------------------------------------

def animate_surface(wave, x, t, *, fps: int = 20, save_path=None):
    """
    Animate the free-surface elevation η(x, t) over the given time array.

    Parameters
    ----------
    wave      : RegularWave or IrregularWave_JONSWAP
    x         : 1-D array of spatial positions [m]
    t         : 1-D array of time values [s]
    fps       : frames per second for display and GIF export
    save_path : path to write a .gif file; None → display only

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)

    # Pre-compute all frames so y-limits are accurate
    frames = [wave.surface_elevation(x, float(ti)) for ti in t]
    eta_max = max(np.max(np.abs(f)) for f in frames) * 1.2 or 1.0

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(-eta_max, eta_max)
    ax.set_xlabel("Distance x [m]")
    ax.set_ylabel(r"Elevation $\eta$ [m]")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    title = ax.set_title("")
    (line,) = ax.plot([], [], "steelblue", lw=1.5)

    def _init():
        line.set_data([], [])
        return line, title

    def _update(i):
        line.set_data(x, frames[i])
        title.set_text(f"Surface elevation  t = {t[i]:.2f} s")
        return line, title

    ani = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=len(t),
        interval=1000 / fps, blit=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer="pillow", fps=fps)
        print(f"Saved: {save_path}")

    return ani


# ---------------------------------------------------------------------------
# Static snapshots
# ---------------------------------------------------------------------------

def snapshot_time(wave, x, t: float, *, ax=None):
    """
    Spatial snapshot of η(x) at a fixed time t.

    Parameters
    ----------
    wave : RegularWave or IrregularWave_JONSWAP
    x    : 1-D array of positions [m]
    t    : time [s]
    ax   : existing Axes to draw on, or None to create a new figure

    Returns
    -------
    fig, ax
    """
    x = np.asarray(x, dtype=float)
    eta = wave.surface_elevation(x, float(t))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(x, eta, color="steelblue", lw=1.5)
    ax.fill_between(x, eta, alpha=0.15, color="steelblue")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Distance x [m]")
    ax.set_ylabel(r"Elevation $\eta$ [m]")
    ax.set_title(f"Spatial snapshot  t = {t:.2f} s")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return fig, ax


def snapshot_space(wave, t, x: float, *, ax=None):
    """
    Time series of η(t) at a fixed spatial position x.

    Parameters
    ----------
    wave : RegularWave or IrregularWave_JONSWAP
    t    : 1-D array of times [s]
    x    : position [m]
    ax   : existing Axes to draw on, or None to create a new figure

    Returns
    -------
    fig, ax
    """
    t = np.asarray(t, dtype=float)
    eta = wave.surface_elevation(float(x), t)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    ax.plot(t, eta, color="k", lw=1.2)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Time t [s]")
    ax.set_ylabel(r"Elevation $\eta$ [m]")
    ax.set_title(f"Time series  x = {x:.1f} m")
    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return fig, ax


# ---------------------------------------------------------------------------
# Energy spectrum
# ---------------------------------------------------------------------------

def plot_spectrum(wave, *, ax=None):
    """
    Energy density spectrum S(f) for a wave object.

    IrregularWave_JONSWAP
        Plots the theoretical JONSWAP target S(f) (red) and the empirical
        spectrum estimated from the generated amplitudes S_emp(f) = a²/(2·Δf)
        (shaded blue). Both in m²/Hz.

    RegularWave
        Plots a single vertical line at f = 1/T representing the discrete
        energy of the monochromatic wave (E = ρgH²/8 [J/m²]).

    Parameters
    ----------
    wave : RegularWave or IrregularWave_JONSWAP
    ax   : existing Axes to draw on, or None to create a new figure

    Returns
    -------
    fig, ax
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.get_figure()

    # IrregularWave_JONSWAP — identified by the presence of amplitude arrays
    if hasattr(wave, "amplitudes") and hasattr(wave, "Hs"):
        df = wave.f[1] - wave.f[0]
        S_target = wave._jonswap()
        S_empirical = wave.amplitudes**2 / (2.0 * df)

        ax.fill_between(wave.f, S_empirical, alpha=0.25, color="steelblue",
                        label="Empirical  $a_i^2 / 2\\Delta f$")
        ax.plot(wave.f, S_empirical, color="steelblue", lw=0.8)
        ax.plot(wave.f, S_target, "r-", lw=1.8, label="JONSWAP target $S(f)$")

        ax.set_xlabel("Frequency f [Hz]")
        ax.set_ylabel(r"$S(f)$  [m² / Hz]")
        ax.set_title(
            f"JONSWAP spectrum  —  Hs = {wave.Hs} m,  Tp = {wave.Tp} s,  γ = {wave.gamma}\n"
            f"target Hs = {wave.Hs:.3f} m   |   actual Hs = {wave.actual_hs:.3f} m"
        )
        ax.legend()

    # RegularWave — identified by scalar H, T, L attributes
    elif hasattr(wave, "H") and hasattr(wave, "L"):
        f0 = 1.0 / wave.T
        E = wave.energy_density  # J/m^2
        ax.axvline(f0, color="steelblue", lw=2.5,
                   label=f"f₀ = {f0:.4f} Hz  (T = {wave.T} s)")
        ax.text(0.62, 0.88, f"E = {E:.1f} J/m²\nH = {wave.H} m",
                transform=ax.transAxes, fontsize=9, color="steelblue",
                va="top", ha="left")
        ax.set_xlim(0, f0 * 3)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Frequency f [Hz]")
        ax.set_ylabel("(monochromatic — single spectral line)")
        ax.set_title(
            f"Regular wave spectrum  —  H = {wave.H} m,  T = {wave.T} s,  d = {wave.d} m"
        )
        ax.legend()

    else:
        raise TypeError(f"Unsupported wave type: {type(wave)!r}")

    ax.grid(True, alpha=0.4)
    fig.tight_layout()

    return fig, ax
