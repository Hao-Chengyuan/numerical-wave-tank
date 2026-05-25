"""
spectrum.py — Wave energy spectral formulations and PSD estimation.

Theoretical spectra
-------------------
pm_spectrum_hs       Pierson-Moskowitz spectrum parameterised by Hs and Tp.
pm_spectrum_classic  Pierson-Moskowitz spectrum in the classic alpha/beta form.
jonswap              JONSWAP spectrum (PM base with peak-enhancement factor).

PSD estimation
--------------
periodogram          One-sided PSD via raw FFT (rectangular window).
avg_periodogram      Averaged periodogram over N equal non-overlapping segments.
welch_psd            Welch's method (Hanning window, configurable overlap).
"""

import numpy as np


def pm_spectrum_hs(f: np.ndarray, Hs: float, Tp: float) -> np.ndarray:
    """
    One-sided Pierson-Moskowitz variance spectrum S(f) [m²/Hz].

    Parameterised by significant wave height Hs and peak period Tp.
    """
    fp = 1.0 / Tp
    return ((5.0 / 16.0) * Hs**2 * fp**4 * f**-5
            * np.exp(-1.25 * (fp / f)**4))


def pm_spectrum_classic(f: np.ndarray, Tp: float, g: float = 9.81) -> np.ndarray:
    """
    One-sided Pierson-Moskowitz variance spectrum S(f) [m²/Hz].

    Uses the original Phillips (1958) constants: alpha=0.0081, beta=1.25.
    """
    alpha = 0.0081
    beta = 1.25
    return alpha * g**2 * (2 * np.pi)**-4 * f**-5 * np.exp(-beta * (1.0 / (Tp * f))**4)


def jonswap(
    f: np.ndarray,
    Hs: float,
    Tp: float,
    gamma: float = 3.3,
    formula: str = "hs",
    g: float = 9.81,
) -> np.ndarray:
    """
    One-sided JONSWAP variance spectrum S(f) [m²/Hz].

    Parameters
    ----------
    f       : frequency array [Hz]; values must be > 0.
    Hs      : significant wave height [m].
    Tp      : peak period [s].
    gamma   : peak-enhancement factor (1 → PM; typical JONSWAP ≈ 3.3).
    formula : 'hs' (default) uses pm_spectrum_hs; 'classic' uses pm_spectrum_classic.
    g       : gravitational acceleration [m/s²].

    Returns
    -------
    S : one-sided variance spectrum [m²/Hz], same shape as f.
    """
    f = np.maximum(f, 1e-6)
    fp = 1.0 / Tp
    sigma = np.where(f <= fp, 0.07, 0.09)

    if formula == "hs":
        S_pm = pm_spectrum_hs(f, Hs, Tp)
    elif formula == "classic":
        S_pm = pm_spectrum_classic(f, Tp, g)
    else:
        raise ValueError("formula must be 'hs' or 'classic'.")

    J = gamma ** np.exp(-0.5 * ((f - fp) / (sigma * fp))**2)
    A_gamma = 1.0 - 0.287 * np.log(gamma)
    return A_gamma * S_pm * J


# ---------------------------------------------------------------------------
# PSD estimation
# ---------------------------------------------------------------------------

def periodogram(segment: np.ndarray, dt: float):
    """
    One-sided PSD estimate via raw FFT (rectangular window).

    Normalisation: S(f) = 2·dt/N · |FFT(x)|²  [m²/Hz]
    The factor of 2 folds negative-frequency energy into the one-sided spectrum.
    DC and Nyquist bins are not doubled (they have no negative-frequency mirror).

    Parameters
    ----------
    segment : 1-D time series (mean-removed recommended).
    dt      : sampling interval [s].

    Returns
    -------
    f : frequency array [Hz], length N//2 + 1.
    S : one-sided PSD [m²/Hz], same length as f.
    """
    N = len(segment)
    X = np.fft.rfft(segment)
    f = np.fft.rfftfreq(N, dt)
    S = (2 * dt / N) * np.abs(X)**2
    S[0] /= 2
    if N % 2 == 0:
        S[-1] /= 2
    return f, S


def avg_periodogram(signal: np.ndarray, dt: float, n_seg: int):
    """
    Averaged periodogram over N equal non-overlapping segments.

    Chops signal into n_seg equal segments, computes a periodogram for each,
    and returns the mean. Frequency resolution: Δf = n_seg / (N·dt).
    Degrees of freedom: 2·n_seg.

    Parameters
    ----------
    signal : 1-D time series.
    dt     : sampling interval [s].
    n_seg  : number of non-overlapping segments.

    Returns
    -------
    f : frequency array [Hz].
    S : averaged one-sided PSD [m²/Hz].
    """
    N  = len(signal)
    Ns = N // n_seg
    segs  = [signal[i * Ns:(i + 1) * Ns] for i in range(n_seg)]
    specs = [periodogram(s, dt)[1] for s in segs]
    f, _  = periodogram(segs[0], dt)
    return f, np.mean(specs, axis=0)


def welch_psd(signal: np.ndarray, dt: float, nperseg: int, noverlap: int | None = None):
    """
    Welch's PSD estimate (Hanning window, configurable overlap).

    Normalisation: S(f) = 2·dt/U · |FFT(w·x_seg)|²  [m²/Hz]
    where U = Σw² is the window power, ensuring the estimate is unbiased.

    Effective degrees of freedom ≈ 1.9 × n_segments for a Hanning window
    at 50 % overlap (Harris, 1978).

    Parameters
    ----------
    signal   : 1-D time series.
    dt       : sampling interval [s].
    nperseg  : samples per segment → frequency resolution Δf = 1 / (nperseg·dt).
    noverlap : overlapping samples between segments (default: nperseg // 2).

    Returns
    -------
    f        : frequency array [Hz], length nperseg//2 + 1.
    S        : Welch one-sided PSD estimate [m²/Hz].
    n_seg    : number of segments averaged.
    """
    if noverlap is None:
        noverlap = nperseg // 2
    step = nperseg - noverlap

    w = np.hanning(nperseg)
    U = np.sum(w**2)
    f = np.fft.rfftfreq(nperseg, dt)

    specs = []
    for start in range(0, len(signal) - nperseg + 1, step):
        seg = signal[start:start + nperseg] * w
        X   = np.fft.rfft(seg)
        S   = (2 * dt / U) * np.abs(X)**2
        S[0] /= 2
        if nperseg % 2 == 0:
            S[-1] /= 2
        specs.append(S)

    return f, np.mean(specs, axis=0), len(specs)
