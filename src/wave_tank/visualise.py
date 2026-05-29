"""
visualise.py — Visualisation utilities for RegularWave and IrregularWave_JONSWAP objects.

Functions
---------
animate_surface(wave, x, t, *, fps=20, save_path=None)
    Animated η(x, t) surface over the given time array.
    Pass x and t as 1-D numpy arrays (e.g. from np.linspace).
    Saves a .gif when save_path is given. Returns the FuncAnimation.

animate_particles(wave, x_positions, z_levels, *, fps=20, n_cycles=2, exaggeration=1.0, save_path=None)
    Airy wave particle orbit animation for a RegularWave.
    Draws a dashed ellipse at each (x, z) grid point and animates a dot tracking
    the instantaneous particle position. `exaggeration` scales orbital radii for
    visual clarity without altering the wave or depth scales.
    Orbits are aspect-ratio corrected so horizontal motion always appears visually
    wider than vertical motion, matching physics regardless of the plot's x/y scale.

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

from spectrum import jonswap


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
        S_target = jonswap(wave.f, wave.Hs, wave.Tp, wave.gamma)
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
                transform=ax.transAxes, color="steelblue",
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


# ---------------------------------------------------------------------------
# Particle orbit animation (RegularWave only)
# ---------------------------------------------------------------------------

def animate_particles(wave, x_positions, z_levels, *,
                      fps: int = 20,
                      n_cycles: int = 2,
                      exaggeration: float = 1.0,
                      save_path=None):
    """
    Animate Airy wave particle orbits under the free surface (RegularWave only).

    At each (x, z) grid point a dashed ellipse shows the full orbital path and
    a coloured dot traces the instantaneous particle position.

    Orbital semi-axes (Airy theory):
        A(z) = a · cosh(k(z+d)) / sinh(kd)   [horizontal]
        B(z) = a · sinh(k(z+d)) / sinh(kd)   [vertical]

    The orbits are rendered with an aspect-ratio correction so that horizontal
    motion appears visually wider than vertical motion regardless of the axes
    scale (the x-axis typically spans hundreds of metres; the z-axis tens).
    Concretely, A is multiplied by (px/m in z) / (px/m in x) so that equal
    physical distances look equal on screen.  Dots track the corrected ellipses,
    so the motion is internally consistent even though the displayed horizontal
    displacement is larger than the raw Airy value.

    Parameters
    ----------
    wave         : RegularWave
    x_positions  : 1-D array of mean particle x-positions [m]
    z_levels     : 1-D array of mean particle depths [m]
                   (0 = SWL, negative downward; must be ≥ −wave.d)
    fps          : frames per second
    n_cycles     : number of wave periods to animate
    exaggeration : multiplicative scale applied to orbital radii for visual
                   clarity. 1.0 = physically accurate; larger values enlarge
                   the orbits without changing the wave or depth axes.
    save_path    : path to write a .gif file; None → display only

    Returns
    -------
    ani : matplotlib.animation.FuncAnimation
    """
    from matplotlib.patches import Ellipse as _Ellipse

    x_positions = np.asarray(x_positions, dtype=float)
    z_levels    = np.asarray(z_levels, dtype=float)

    k, d, a = wave.k, wave.d, wave.a
    sinh_kd = np.sinh(k * d)

    # Physical orbital semi-axes with user exaggeration
    A_phys = exaggeration * a * np.cosh(k * (z_levels + d)) / sinh_kd  # (nz,)
    B_phys = exaggeration * a * np.sinh(k * (z_levels + d)) / sinh_kd  # (nz,)

    n_frames = max(int(n_cycles * wave.T * fps), 2)
    t_array  = np.linspace(0, n_cycles * wave.T, n_frames, endpoint=False)

    z_top   = a * 2.5
    z_bot   = -d * 1.08
    z_range = z_top - z_bot
    x_span  = float(np.ptp(x_positions)) if len(x_positions) > 1 else 0.0

    # --- Pass 1: initial layout with A_phys-based padding ---
    x_pad_0 = max(float(A_phys[0]) * 1.5, x_span * 0.05 + 0.5)
    x_min   = float(x_positions[0])  - x_pad_0
    x_max   = float(x_positions[-1]) + x_pad_0

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(z_bot, z_top)
    ax.set_xlabel("Distance x [m]")
    ax.set_ylabel("Elevation z [m]")

    exag_tag = f"  (orbits ×{exaggeration:.0f})" if exaggeration != 1.0 else ""
    ax.set_title(
        f"Particle orbits — H = {wave.H} m,  T = {wave.T} s,  d = {wave.d} m{exag_tag}"
    )
    fig.tight_layout()

    # Derive pixels-per-data-unit from the finalised axes box.
    # ax.get_position() is updated by tight_layout without needing canvas.draw().
    pos     = ax.get_position()
    fig_w, fig_h = fig.get_size_inches()
    ax_w_px = pos.width  * fig_w * fig.dpi
    ax_h_px = pos.height * fig_h * fig.dpi
    # Scale horizontal semi-axis so 1 m in x occupies the same screen pixels
    # as 1 m in z — this makes A visually larger than B whenever A > B (always true).
    horiz_scale = (ax_h_px / z_range) / (ax_w_px / (x_max - x_min))

    A_vis = A_phys * horiz_scale  # (nz,) — aspect-corrected horizontal semi-axis
    B_vis = B_phys                # (nz,) — vertical semi-axis unchanged

    # --- Pass 2: expand x limits if visual orbits exceed initial padding ---
    x_pad = max(float(A_vis[0]) * 1.5, x_span * 0.05 + 0.5)
    if x_pad > x_pad_0 + 0.01:
        x_min = float(x_positions[0])  - x_pad
        x_max = float(x_positions[-1]) + x_pad
        ax.set_xlim(x_min, x_max)
        fig.tight_layout()
        pos     = ax.get_position()
        ax_w_px = pos.width * fig_w * fig.dpi
        horiz_scale = (ax_h_px / z_range) / (ax_w_px / (x_max - x_min))
        A_vis = A_phys * horiz_scale

    x_surf = np.linspace(x_min, x_max, 400)

    # --- Static background ---
    ax.axhspan(-d, 0, color="lightcyan", alpha=0.5, zorder=1)
    ax.fill_between([x_min, x_max], z_bot, -d,
                    color="saddlebrown", alpha=0.65, zorder=2)
    ax.axhline(-d, color="sienna", lw=1.5, zorder=3)
    ax.axhline(0, color="navy", lw=0.7, ls="--", alpha=0.4,
               zorder=3, label="SWL")
    ax.legend(loc="upper right")

    # --- Aspect-corrected orbit ellipses ---
    for xi in x_positions:
        for j, zj in enumerate(z_levels):
            ax.add_patch(_Ellipse(
                xy=(float(xi), float(zj)),
                width=2 * float(A_vis[j]), height=2 * float(B_vis[j]),
                fill=False, edgecolor="steelblue", lw=1.0,
                linestyle="--", alpha=0.55, zorder=4,
            ))

    # --- Animated artists ---
    surf_line, = ax.plot([], [], color="royalblue", lw=2.0, zorder=6)
    dots, = ax.plot([], [], "o", color="orangered", ms=7, zorder=7,
                    markeredgecolor="firebrick", markeredgewidth=0.7)
    time_text = ax.text(0.01, 0.97, "", transform=ax.transAxes,
                        va="top", ha="left")

    def _init():
        surf_line.set_data([], [])
        dots.set_data([], [])
        time_text.set_text("")
        return surf_line, dots, time_text

    def _update(frame_i):
        ti = t_array[frame_i]
        surf_line.set_data(x_surf, wave.surface_elevation(x_surf, ti))

        # Phase at each mean position x₀
        theta_0 = wave.k * x_positions - wave.omega * ti + wave.phase  # (nx,)

        # Horizontal displacement uses A_vis (aspect-corrected)
        xi_disp = -A_vis[np.newaxis, :] * np.sin(theta_0[:, np.newaxis])  # (nx, nz)
        x_pts   =  x_positions[:, np.newaxis] + xi_disp                    # (nx, nz)

        # Evaluate vertical phase at the *displayed* x, not the mean position.
        # This ensures surface particles align exactly with the surface wave line
        # even though x is visually exaggerated by horiz_scale.
        theta_at_x = wave.k * x_pts - wave.omega * ti + wave.phase  # (nx, nz)
        zeta_disp  = B_vis[np.newaxis, :] * np.cos(theta_at_x)      # (nx, nz)

        dots.set_data(x_pts.ravel(), (z_levels[np.newaxis, :] + zeta_disp).ravel())
        time_text.set_text(f"t = {ti:.2f} s")
        return surf_line, dots, time_text

    ani = animation.FuncAnimation(
        fig, _update, init_func=_init, frames=n_frames,
        interval=1000 / fps, blit=True,
    )

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(save_path), writer="pillow", fps=fps)
        print(f"Saved: {save_path}")

    return ani
