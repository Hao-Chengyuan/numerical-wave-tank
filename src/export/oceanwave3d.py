"""
oceanwave3d.py — Export IrregularWave_JONSWAP eta(t) as an OceanWave3D
.iwf (custom incident wave) file.

OceanWave3D consumes this file when OceanWave3D.inp line 22 sets i_spec=2.
Format:
  Line 1: free-form descriptive header (ignored by the solver)
  Line 2: dt in seconds
  Lines 3..N+2: surface elevation eta(t) values, one per line, uniformly
                spaced at dt; SI units (m, s).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import numpy as np

from wave_tank import IrregularWave_JONSWAP


def write_iwf(
    path: Union[str, Path],
    wave: IrregularWave_JONSWAP,
    x: float,
    t: np.ndarray,
    *,
    header: Union[str, None] = None,
    fmt: str = "%.6f",
) -> None:
    """Write eta(t) at probe x to an OceanWave3D .iwf file.

    Parameters
    ----------
    path : str or Path
        Output file path.
    wave : IrregularWave_JONSWAP
        Sea state to sample.
    x : float
        Probe location in metres.
    t : np.ndarray
        1-D array of times in seconds, uniformly spaced (>= 2 samples).
    header : str, optional
        Descriptive line 1. Auto-generated from wave parameters if None.
    fmt : str, optional
        printf-style format for eta values. Default "%.6f".
    """
    if not isinstance(wave, IrregularWave_JONSWAP):
        raise TypeError(
            f"wave must be an IrregularWave_JONSWAP, got {type(wave).__name__}"
        )

    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size < 2:
        raise ValueError("t must be a 1-D array with at least 2 samples")

    diffs = np.diff(t)
    dt = float(diffs[0])
    if not np.all(np.abs(diffs - dt) < 1e-9):
        raise ValueError("t must have uniform spacing within 1e-9 tolerance")

    eta = wave.surface_elevation(float(x), t)

    if header is None:
        header = (
            f"JONSWAP Hs={wave.Hs:.6f} Tp={wave.Tp:.6f} "
            f"gamma={wave.gamma:.3f} d={wave.d:.3f} "
            f"x={float(x):.6f} N={t.size}"
        )

    path = Path(path)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with open(tmp_path, "w") as f:
            f.write(header + "\n")
            f.write(f"{dt:.6f}\n")
            for value in eta:
                f.write((fmt + "\n") % value)
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
