# Code Review: `src/` — Physics & Code Quality

---

## 1. Physics / Mathematical Reproduction

### 1.1 `jonswap()` — `Hs` is silently ignored when `formula='classic'`

**File:** [spectrum.py:70-75](src/spectrum.py#L70-L75)

```python
elif formula == "classic":
    S_pm = pm_spectrum_classic(f, Tp, g)   # Hs is never passed
```

`pm_spectrum_classic` uses the universal Phillips constant α = 0.0081 to set the energy level; it has no `Hs` parameter. When a caller writes `jonswap(f, Hs=3.0, Tp=10.0, formula='classic')` the target Hs is silently discarded and the returned spectrum has whatever energy the α constant implies for that peak period. This is the most significant API issue in the codebase — add a `warnings.warn` or raise a `ValueError` clarifying that `formula='classic'` is incompatible with a target Hs.

---

### 1.2 `jonswap()` — No bounds check on `gamma`; `A_gamma` can go negative

**File:** [spectrum.py:78](src/spectrum.py#L78)

```python
A_gamma = 1.0 - 0.287 * np.log(gamma)
```

The Goda (1985) normalisation constant is only an empirical fit for γ ∈ [1, 7]. For γ > e^(1/0.287) ≈ 32.5 the constant becomes negative, yielding a physically impossible spectrum with negative energy density. Practical JONSWAP uses γ ≈ 1–7, but the code accepts any positive float without complaint. A guard like `if not (1.0 <= gamma <= 7.0): warnings.warn(...)` would prevent silent nonsense.

---

### 1.3 `pm_spectrum_classic` — Docstring misattributes constants

**File:** [spectrum.py:32-39](src/spectrum.py#L32-L39)

The docstring reads *"Uses the original Phillips (1958) constants: alpha=0.0081, beta=1.25."* Phillips (1958) proposed the f⁻⁵ equilibrium-range form and the α = 0.0081 constant, but the beta = 1.25 peak coefficient in the fully-developed form comes from Pierson & Moskowitz (1964). The function is implementing the P-M (1964) spectrum, not the Phillips (1958) range.  A more accurate description: *"Pierson-Moskowitz (1964) fully-developed spectrum, α = 0.0081 (Phillips 1958), β = 1.25."*

---

### 1.4 `animate_particles` — Vertical phase evaluated at displaced position (Airy inconsistency)

**File:** [visualise.py:422-423](src/visualise.py#L422-L423)

```python
theta_at_x = wave.k * x_pts - wave.omega * ti + wave.phase   # x_pts = x₀ + ξ_disp
zeta_disp  = B_vis[np.newaxis, :] * np.cos(theta_at_x)
```

Airy (linear) theory evaluates all kinematics at the *mean* particle position x₀, not the displaced position. Evaluating the vertical displacement at `x_pts = x₀ + ξ` implicitly blends in second-order Stokes corrections in an ad hoc way. The docstring does explain this is done so surface particles visually align with the wave crest, but the behaviour should be marked as a visual approximation that departs from the underlying linear model. Consider a flag like `align_to_surface: bool = True` so users can toggle the strict Airy behaviour.

---

### 1.5 Seabed boundary condition not enforced at z = –d (divide-by-zero risk)

**File:** [wave_tank.py:285-286](src/wave_tank.py#L285-L286), [wave_tank.py:132-137](src/wave_tank.py#L132-L137)

```python
cosh_f = np.cosh(self.k * (z + self.d)) / np.sinh(self.k * self.d)
```

At z = –d the numerator `cosh(0) = 1` and denominator `sinh(0) = 0`, producing infinity (or NaN after a divide-by-zero warning). The code never validates that z ∈ [−d, 0], so a caller passing z = −d exactly will get silent bad output. The same issue exists in `RegularWave._cosh_ratio` and `_sinh_ratio`. A simple `if np.any(z < -self.d) or np.any(z > 0): warnings.warn(...)` and a special-case for z = –d (where w = 0 exactly) would fix both.

---

### 1.6 `IrregularWave_JONSWAP` — Amplitude Rayleigh scale and target Hs

**File:** [wave_tank.py:220-221](src/wave_tank.py#L220-L221)

```python
scale = np.sqrt(S_f * df)
self.amplitudes = rng.rayleigh(scale=scale)
```

The Tucker et al. (1984) Rayleigh method gives E[aᵢ²] = 2σᵢ² = 2·S(f)·Δf, so E[m₀] = Σ S(f)·Δf ≈ Hs²/16 (guaranteed by the Goda normalisation). This is statistically correct, but individual realisations scatter around the target. The 15% tolerance in [fun_test.py:85](src/fun_test.py#L85) reflects that reality. One lesser-known implication: Rayleigh-distributed amplitudes produce a surface elevation that is *not* Gaussian (the phase randomness makes it approximately Gaussian for large N, but the amplitude distribution adds excess kurtosis for small N). Worth a brief comment.

---

### 1.7 `IrregularWave_JONSWAP` — High-frequency spectral tail truncation

**File:** [wave_tank.py:210](src/wave_tank.py#L210)

```python
self.f = np.arange(f_min_factor * fp, f_max_factor * fp, df)
```

Default range [0.5fp, 5fp]. Below 0.5fp the JONSWAP f⁻⁵ term is killed by the exponential (≈ e⁻²⁰ at 0.5fp), so the low-frequency truncation is inconsequential. Above 5fp the spectrum enters its power-law tail; the truncated fraction is approximately (1/5)⁴ ≈ 0.16% of total energy — also negligible. Both limits are defensible and worth a brief inline note explaining why they are safe.

---

## 2. Code Quality / Potential Optimisations

### 2.1 `avg_periodogram` — `periodogram` called twice on the first segment

**File:** [spectrum.py:135-138](src/spectrum.py#L135-L138)

```python
specs = [periodogram(s, dt)[1] for s in segs]   # index 0 already computed here
f, _  = periodogram(segs[0], dt)                 # duplicate call
```

The first segment's periodogram is computed twice. One fix:

```python
results = [periodogram(s, dt) for s in segs]
f = results[0][0]
return f, np.mean([r[1] for r in results], axis=0)
```

---

### 2.2 `welch_psd` — Python loop; vectorisable with stride tricks

**File:** [spectrum.py:172-182](src/spectrum.py#L172-L182)

The per-segment loop is not a bottleneck for typical signal lengths, but it can be replaced with a single 2-D FFT using `np.lib.stride_tricks.sliding_window_view`:

```python
view = np.lib.stride_tricks.sliding_window_view(signal, nperseg)[::step]  # (n_seg, nperseg)
segments = view * w                                                          # broadcast Hann window
X = np.fft.rfft(segments, axis=1)
S_all = (2 * dt / U) * np.abs(X)**2
S_all[:, 0] /= 2
if nperseg % 2 == 0:
    S_all[:, -1] /= 2
return f, S_all.mean(axis=0), len(view)
```

This also moves the DC/Nyquist correction out of the loop.

---

### 2.3 `solve_dispersion` — relative tolerance can stall near zero

**File:** [wave_tank.py:48](src/wave_tank.py#L48)

```python
if np.all(np.abs(dk) < tol * np.abs(k)):
```

If k is very small (very long waves, shallow water), `tol * |k|` approaches the floating-point floor and the loop may iterate all 50 times without triggering the absolute break condition. Adding an absolute floor — `tol * max(|k|, 1e-12)` — is a common defence:

```python
if np.all(np.abs(dk) < tol * np.maximum(np.abs(k), 1e-12)):
```

---

### 2.4 `IrregularWave_JONSWAP.particle_velocity` — scalar-only contract unenforced

**File:** [wave_tank.py:278-290](src/wave_tank.py#L278-L290)

The docstring says *"All three arguments must be scalars"* but nothing in the code enforces this. Passing an array for z silently changes the shape of `cosh_f` from (N,) to (N, M) and the dot products produce an unexpected result (or raise a shape error). A guard at the top of the method would catch this early.

---

### 2.5 G and RHO duplicated across both wave classes

**File:** [wave_tank.py:63-64](src/wave_tank.py#L63-L64) and [wave_tank.py:186-187](src/wave_tank.py#L186-L187)

`G = 9.81` and `RHO = 1025.0` are defined identically on both `RegularWave` and `IrregularWave_JONSWAP`. A single pair of module-level constants at the top of `wave_tank.py` (and mirrored in `spectrum.py` where `g` is a parameter default) would be a single source of truth.

---

### 2.6 `surface_elevation` (IrregularWave) — large intermediate arrays

**File:** [wave_tank.py:258-262](src/wave_tank.py#L258-L262)

```python
phase = (self.k[:, np.newaxis] * x[np.newaxis, :]
         - (self.omega * float(t))[:, np.newaxis]
         + self.phases[:, np.newaxis])
return np.einsum('n,nm->m', self.amplitudes, np.cos(phase))
```

The `phase` and `np.cos(phase)` intermediates are each of shape (N_components × N_x) — e.g., 500 × 10 000 × 8 bytes ≈ 40 MB per array for a large spatial grid. For typical notebook use this is fine, but for long time-stepping loops it adds up. `np.einsum` supports `optimize=True` which can reorder contractions; alternatively, chunking over x in blocks of ~1 000 keeps peak memory < 4 MB per call.

---

### 2.7 `plot_spectrum` — fragile duck-typing

**File:** [visualise.py:223-242](src/visualise.py#L223-L242)

```python
if hasattr(wave, "amplitudes") and hasattr(wave, "Hs"):
    ...
elif hasattr(wave, "H") and hasattr(wave, "L"):
    ...
```

This breaks silently if either class grows or a third wave type is added. Using `isinstance(wave, IrregularWave_JONSWAP)` and `isinstance(wave, RegularWave)` is unambiguous, self-documenting, and raises an error at the right level when an unexpected type is passed.

---

### 2.8 `IrregularWave_JONSWAP.__init__` — `np.arange` with float step

**File:** [wave_tank.py:210](src/wave_tank.py#L210)

```python
self.f = np.arange(f_min_factor * fp, f_max_factor * fp, df)
```

`np.arange` with floating-point start/step can include or exclude the last value unpredictably due to rounding. The NumPy docs recommend `np.linspace` for float ranges. Here the exact number of components is `round((f_max - f_min) / df)`, which can be computed first and then passed to `np.linspace`:

```python
n_comp = round((f_max_factor - f_min_factor) * fp / df)
self.f = np.linspace(f_min_factor * fp, f_max_factor * fp, n_comp, endpoint=False)
```

---

### 2.9 `animate_surface` — all frames pre-computed into memory

**File:** [visualise.py:81](src/visualise.py#L81)

```python
frames = [wave.surface_elevation(x, float(ti)) for ti in t]
```

Pre-computing every frame is needed to set accurate y-limits, but for a 3-hour animation at 20 fps this is 216 000 frames. For long `t` arrays the list can consume hundreds of MB. One approach: do a single pass to find `eta_max` (calling `surface_elevation` for each t but not storing), then use a generator in `_update` for the animation itself.

---

### 2.10 `RegularWave._cosh_ratio` / `_sinh_ratio` — redundant `sinh(kd)` evaluation

**File:** [wave_tank.py:131-137](src/wave_tank.py#L131-L137)

Both helpers recompute `np.sinh(self.k * self.d)` independently. The value is constant for a given wave and could be cached once (e.g., as `self._sinh_kd` in `__init__`). In a hot loop (time-stepping over many t values) this saves two transcendental function calls per kinematics call.

---

## Summary Table

| # | Location | Category | Severity |
|---|----------|----------|----------|
| 1.1 | `spectrum.py:70` | Physics / API | High |
| 1.2 | `spectrum.py:78` | Physics | Medium |
| 1.3 | `spectrum.py:32` | Docs | Low |
| 1.4 | `visualise.py:422` | Physics (visual) | Low |
| 1.5 | `wave_tank.py:285`, `132` | Physics / Robustness | Medium |
| 1.6 | `wave_tank.py:220` | Physics (doc) | Low |
| 1.7 | `wave_tank.py:210` | Physics (doc) | Low |
| 2.1 | `spectrum.py:135` | Optimisation | Low |
| 2.2 | `spectrum.py:172` | Optimisation | Low |
| 2.3 | `wave_tank.py:48` | Robustness | Low |
| 2.4 | `wave_tank.py:278` | Robustness | Medium |
| 2.5 | `wave_tank.py:63`, `186` | Maintainability | Low |
| 2.6 | `wave_tank.py:258` | Memory | Low |
| 2.7 | `visualise.py:223` | Maintainability | Low |
| 2.8 | `wave_tank.py:210` | Correctness (edge) | Low |
| 2.9 | `visualise.py:81` | Memory | Low |
| 2.10 | `wave_tank.py:131` | Optimisation | Low |
