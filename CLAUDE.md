# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.12 virtualenv at `.venv/`. Always use `.venv/bin/python` (or `uv run python`) to run scripts.

```bash
.venv/bin/python src/fun_test.py       # run all validation tests and benchmarks
.venv/bin/jupyter notebook             # open notebooks (launch from project root)
```

Key packages: `numpy 2.4`, `matplotlib 3.10`, `pillow` (GIF export), `ipykernel` (notebooks). `scipy` is **not** installed.

## Repository layout

```
numerical-wave-tank/
├── src/
│   ├── wave_tank.py     # solve_dispersion, RegularWave, IrregularWave_JONSWAP
│   ├── spectrum.py      # pm_spectrum_hs, pm_spectrum_classic, jonswap,
│   │                    #   periodogram, avg_periodogram, welch_psd
│   ├── visualise.py     # plot_spectrum, snapshot_time, snapshot_space,
│   │                    #   animate_surface, animate_particles
│   ├── fun_test.py      # validation tests and performance benchmarks
│   └── wave_gen.py      # original reference implementation (do not modify)
├── Notebooks/
│   ├── Wave_Tank_Visualise.ipynb          # exercises all visualise.py functions
│   ├── JONSWAP_Spectral_Verification.ipynb # spectral estimation study
│   └── Wave_Simulation_Comparison.ipynb   # self-contained Tucker et al. study
├── output/              # generated GIFs; not tracked by git
├── .venv/               # virtual environment (recreated via uv sync)
├── pyproject.toml
└── uv.lock
```

## Notebook path convention

Notebooks live in `Notebooks/`. Jupyter sets the kernel CWD to the notebook directory, so imports must use relative paths:

```python
import sys
sys.path.insert(0, '../src')   # correct — resolves to project src/

from wave_tank import RegularWave, IrregularWave_JONSWAP
from spectrum import jonswap, periodogram, welch_psd
```

Output files (GIFs) are written to `../output/` for the same reason.

## Architecture

**`wave_tank.py`** — wave generation and kinematics

- `solve_dispersion(omega, d)` — vectorized Newton-Raphson dispersion solver
- `RegularWave` — first-order Airy wave with kinematics, energy, and dynamic pressure
- `IrregularWave_JONSWAP` — JONSWAP sea state built from many `RegularWave` components; delegates spectrum construction to `spectrum.jonswap()`

**`spectrum.py`** — spectral formulations and PSD estimation

- `pm_spectrum_hs(f, Hs, Tp)` — Pierson-Moskowitz spectrum parameterised by Hs/Tp
- `pm_spectrum_classic(f, Tp)` — PM in original Phillips (1958) alpha/beta form
- `jonswap(f, Hs, Tp, gamma)` — JONSWAP = PM × peak-enhancement factor with Goda normalisation
- `periodogram(segment, dt)` — one-sided raw FFT PSD (rectangular window)
- `avg_periodogram(signal, dt, n_seg)` — averaged periodogram over N equal non-overlapping segments
- `welch_psd(signal, dt, nperseg, noverlap)` — Welch's method with Hanning window and configurable overlap

**`visualise.py`** — matplotlib helpers for notebooks

## Key physics conventions

- Coordinate: `z` measured **upward** from still-water level; seabed is `z = -d`.
- Phase convention: `η = a·cos(kx − ωt + φ)`.
- Hs verification: `actual_hs = 4·√(m₀)` where `m₀ = Σ(aᵢ²)/2`.
- Breaking checks (warnings, not errors): steepness `H/L > 1/7`, depth-limited `H/d > 0.78`.
- `RHO = 1025.0 kg/m³`, `G = 9.81 m/s²` are class-level constants on both wave classes.

## Dispersion solver

`solve_dispersion(omega, d)` in `wave_tank.py` is the canonical solver. It uses Hunt (1979) for the initial guess and converges in ≤ 50 Newton-Raphson iterations with tolerance `1e-10`. Pass an array of `omega` values to solve all components in one vectorised call (137× faster than solving per-component).

## IrregularWave_JONSWAP design

Components are stored as flat numpy arrays (`self.k`, `self.omega`, `self.amplitudes`, `self.phases`) instead of per-component Python objects. `surface_elevation` uses broadcasting + `np.einsum` to evaluate all components simultaneously. The JONSWAP spectrum is constructed by calling `spectrum.jonswap()`, applying the Goda normalisation constant `A_γ = 1 − 0.287·ln(γ)` to preserve the target Hs. Amplitudes are Rayleigh-distributed (Tucker et al., 1984).

## Notebooks

| Notebook | Purpose |
|---|---|
| `Wave_Tank_Visualise.ipynb` | Exercises `visualise.py`: spectrum plot, spatial/temporal snapshots, surface animation, particle orbit animation |
| `JONSWAP_Spectral_Verification.ipynb` | Spectral verification study: raw periodogram, fixed-segment averaging, Welch's method vs target JONSWAP |
| `Wave_Simulation_Comparison.ipynb` | Self-contained Tucker et al. (1984) study of deterministic vs Rayleigh amplitude methods (no imports from `src/`) |
