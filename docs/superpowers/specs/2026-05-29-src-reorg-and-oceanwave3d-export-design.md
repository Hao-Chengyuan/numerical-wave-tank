# src/ Reorganization and OceanWave3D Export — Design

Date: 2026-05-29

## Goal

Reorganize `src/` into topical Python sub-packages and add a new exporter that
writes `IrregularWave_JONSWAP` surface-elevation time series in OceanWave3D's
`.iwf` (custom incident-wave) format.

The reorg is motivated by needing a dedicated home for an OceanWave3D exporter
and by the fact that `src/` had grown to mix wave generation, spectral tools,
visualisation, and validation tests as flat modules.

## New `src/` layout

```
src/
├── wave_tank/
│   ├── __init__.py          # re-exports RegularWave, IrregularWave_JONSWAP, solve_dispersion
│   ├── wave_tank.py         # moved as-is
│   └── visualise.py         # moved here (kept as a submodule, not re-exported)
├── spectrum/
│   ├── __init__.py          # re-exports jonswap, pm_spectrum_hs, pm_spectrum_classic,
│   │                        #   periodogram, avg_periodogram, welch_psd
│   └── spectrum.py          # moved as-is
├── tests/
│   ├── __init__.py
│   └── fun_test.py          # moved
└── export/
    ├── __init__.py          # re-exports write_iwf
    └── oceanwave3d.py       # NEW
```

Each sub-package follows the same pattern: an `__init__.py` re-exports the
public API of the implementation module so that existing flat imports keep
working after the directory move. `visualise` is the one exception: it is
exposed as a submodule of `wave_tank` (`wave_tank.visualise`) rather than
re-exported at the top level, to keep plotting separate from physics in the
namespace.

## Imports

Notebooks keep their single `sys.path.insert(0, '../src')`. The test script
uses a `pathlib`-based shim to add `src/` to `sys.path` regardless of CWD
(see "tests/fun_test.py changes" below).

After the reorg the import surface is:

```python
from wave_tank import RegularWave, IrregularWave_JONSWAP, solve_dispersion
from wave_tank.visualise import plot_spectrum, animate_surface, snapshot_time, ...
from spectrum import jonswap, welch_psd, periodogram, ...
from export.oceanwave3d import write_iwf
```

Only one user-facing churn:

- Notebooks that currently do `from visualise import …` must change to
  `from wave_tank.visualise import …`.

Cross-module imports inside `src/` keep working unchanged because `src/` stays
on `sys.path`, so `from spectrum import jonswap` inside
`src/wave_tank/wave_tank.py` continues to resolve.

## OceanWave3D `.iwf` format reference

Observed from
`OceanWave3D Doc/input file example/Wave_generation/Wave_Gen_customise_input/examples/IncidentWave.iwf`:

- Line 1: free-form descriptive header (the solver does not parse it).
- Line 2: `dt` in seconds, as a plain float on its own line.
- Lines 3..N+2: surface elevation η(t) values in metres, one per line, sampled
  at uniform spacing `dt`. The total number of samples must be at least
  `Nsteps + 1` where `Nsteps` is the value declared in `OceanWave3D.inp` line 5.

Wave parameters used by the OceanWave3D solver (T_p, H_s, h_0, x_0, y_0) come
from `OceanWave3D.inp` line 22, not from the `.iwf` file. The `.iwf` only
supplies the time series.

Units: SI (m, s). Matches the conventions already used by
`wave_tank.RegularWave` and `wave_tank.IrregularWave_JONSWAP`.

## `export.oceanwave3d.write_iwf` API

```python
def write_iwf(
    path: str | Path,
    wave: IrregularWave_JONSWAP,
    x: float,
    t: np.ndarray,
    *,
    header: str | None = None,
    fmt: str = "%.6f",
) -> None
```

Behaviour:

1. Validate `t` is a 1-D numpy array with at least 2 samples and uniform
   spacing within `1e-9` (absolute tolerance on consecutive diffs). Otherwise
   raise `ValueError`.
2. Validate `wave` is an instance of `IrregularWave_JONSWAP`. Otherwise raise
   `TypeError`. (`RegularWave` is explicitly out of scope for this iteration.)
3. Compute `eta = wave.surface_elevation(x, t)` — uses the existing
   fixed-x / array-t code path in `IrregularWave_JONSWAP.surface_elevation`.
4. Compute `dt = float(t[1] - t[0])`.
5. Build the header line:
   - If `header` is provided, use it verbatim.
   - Otherwise auto-generate:
     `f"JONSWAP Hs={wave.Hs:.6f} Tp={wave.Tp:.6f} gamma={wave.gamma:.3f} d={wave.d:.3f} x={x:.6f} N={len(t)}"`
6. Write the file:
   - Line 1: header
   - Line 2: `f"{dt:.6f}"`
   - Lines 3..N+2: each `eta[i]` formatted with `fmt`, one per line, trailing
     newline on each line including the last.
7. No conversion of units. The function assumes SI input.

The function returns `None`; it raises on validation failure and otherwise
writes the file atomically (write to a sibling temp path, then `os.replace`)
so a partial write does not corrupt an existing file.

## `tests/fun_test.py` changes

1. Add a 2-line `sys.path` shim at the top so the script keeps running
   standalone via `.venv/bin/python src/tests/fun_test.py`:

   ```python
   import sys, pathlib
   sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
   ```

2. Add one new test: `test_write_iwf_roundtrip`.
   - Build a small `IrregularWave_JONSWAP(Hs=2.0, Tp=8.0, d=30.0, seed=0)`.
   - Generate `t = np.arange(0, 60, 0.1)`.
   - Call `write_iwf(tmp_path, wave, x=0.0, t=t)`.
   - Read the file back, verify:
     - Line 1 starts with `"JONSWAP "`.
     - Line 2 parses to `dt == 0.1` within `1e-9`.
     - Remaining lines: `len == len(t)`, first sample matches
       `wave.surface_elevation(0.0, t)[0]` within `5e-7` (rounding from `%.6f`).

## CLAUDE.md updates

- Rewrite the "Repository layout" tree to match the new layout.
- Update the "Notebook path convention" code block to add the
  `from wave_tank.visualise import …` form.
- Add a short bullet under "Architecture" describing `export.oceanwave3d` and
  the `.iwf` format.

## Out of scope (deferred, flagged here for future)

- Splitting `fun_test.py` into multiple per-module test files. The new folder
  structure supports it but no need to do it in this change.
- `write_iwf` support for `RegularWave`. Trivial to add later by computing
  `eta = wave.surface_elevation(x, t)` via the regular-wave path.
- Reading an `.iwf` file back as a public API (the round-trip test does its own
  ad-hoc parsing; no `read_iwf` is exposed).
- Storing the JONSWAP seed on the `IrregularWave_JONSWAP` instance so the
  auto-header can include it. Currently `seed` is consumed only by the local
  RNG and is not retained.
