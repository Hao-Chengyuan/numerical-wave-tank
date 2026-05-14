# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Python 3.12 virtualenv at `.venv/`. Always use `.venv/bin/python` (or `uv run python`) to run scripts.

```bash
.venv/bin/python src/fun_test.py       # run all validation tests and benchmarks
.venv/bin/jupyter notebook             # open the comparison notebook
```

Key packages: `numpy 2.4`, `matplotlib 3.10`, `pillow` (GIF export), `ipykernel` (notebooks). `scipy` is **not** installed.

## Repository layout

```
numerical-wave-tank/
├── src/                        # all Python source files
│   ├── wave_gen_new.py         # main module: solve_dispersion, RegularWave, IrregularWave_JONSWAP
│   ├── fun_test.py             # validation tests and performance benchmarks
│   └── wave_gen.py             # original reference implementation
├── output/                     # generated files (GIFs); not tracked by git
├── .venv/                      # virtual environment (recreated via uv sync)
├── Wave_Simulation_Comparison.ipynb
├── pyproject.toml
└── uv.lock
```

## Architecture

Both wave modules share the same two-layer design:

**Layer 1 — `RegularWave`**: a single deterministic Airy (linear, first-order) wave defined by height `H`, period `T`, depth `d`, and optional phase. Initialisation solves the dispersion relation `ω² = gk·tanh(kd)` via Newton-Raphson. In `src/wave_gen_new.py` this is delegated to the standalone `solve_dispersion(omega, d)` which accepts scalar or array inputs.

**Layer 2 — `IrregularWave_JONSWAP`**: a random sea state built by superimposing many `RegularWave` components whose amplitudes are Rayleigh-distributed (Tucker et al., 1984) and whose frequencies span `[f_min_factor·fp, f_max_factor·fp]` in steps of `1/T_repeat`. The JONSWAP spectrum (`_jonswap()`) applies the Goda normalization constant `A_γ = 1 − 0.287·ln(γ)` to preserve the target `Hs`.

**Critical design decision in `src/wave_gen_new.py`**: components are stored as flat numpy arrays (`self.k`, `self.omega`, `self.amplitudes`, `self.phases`) instead of per-component Python objects. `surface_elevation` uses broadcasting + `np.einsum` to evaluate all components simultaneously, avoiding a Python loop.

## Key physics conventions

- Coordinate: `z` measured **upward** from still-water level; seabed is `z = -d`.
- Phase convention: `η = a·cos(kx − ωt + φ)`.
- Hs verification: `actual_hs = 4·√(m₀)` where `m₀ = Σ(aᵢ²)/2`.
- Breaking checks (warnings, not errors): steepness `H/L > 1/7`, depth-limited `H/d > 0.78`.
- `RHO = 1025.0 kg/m³`, `G = 9.81 m/s²` are class-level constants on both wave classes.

## Dispersion solver

`solve_dispersion(omega, d)` in `src/wave_gen_new.py` is the canonical solver. It uses Hunt (1979) for the initial guess and converges in ≤ 50 Newton-Raphson iterations with tolerance `1e-10`. Pass an array of `omega` values to solve all components in one vectorised call (137× faster than solving per-component).

## The notebook

`Wave_Simulation_Comparison.ipynb` is a self-contained study (no imports from `wave_gen*.py`). It documents the Tucker et al. (1984) finding that deterministic amplitudes `Aᵢ = √(2S(fᵢ)Δf)` suppress realistic wave groupiness, while Rayleigh-drawn amplitudes produce a proper Gaussian sea surface. Method 1 (deterministic) always realises exactly the target `Hs`; Method 2 (Rayleigh) scatters around it as real ocean records do.
