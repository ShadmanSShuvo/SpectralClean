# SpectralClean v2 — Implementation Walkthrough

## What Was Built

All 4 phases from `plan-v2.md` are implemented and passing tests.

---

## Phase 1 — `effects.py` [NEW]

A new standalone DSP module with four effect families, all implemented using only NumPy (no `scipy.signal.*`).

### Equalizer (IIR Biquad)
- 6 biquad filter types: `lowpass`, `highpass`, `peaking`, `lowshelf`, `highshelf`
- Coefficients derived from the Audio EQ Cookbook formulae — manually coded
- Difference equation runs as a sample-by-sample Python loop (`_biquad_filter`)
- `Equalizer.graphic_eq(gains_db, fs)` convenience constructor maps 5 gains to the standard bands `[60, 230, 910, 4000, 14000]` Hz

### Echo / Delay
- Feedback comb filter: `y[n] = x[n] + g·y[n−M]`
- `M = round(delay_ms / 1000 * fs)` samples; `g` enforced `< 1.0`
- Dry/wet mix parameter

### Algorithmic Reverb (`apply_reverb`)
- Schroeder/Freeverb network: 4 parallel comb filters with low-pass damping → 2 series all-pass diffusers
- `_comb_filter` and `_allpass_filter` both use manual Python sample loops
- Configurable `room_size`, `damping`, `wet`

### Convolution Reverb (`apply_convolution_reverb`)
- Convolves input with an uploaded RIR using existing `convolve_fast` from `convolution.py`
- Normalises the RIR to unit peak; trims output to original length

### Spectral Subtraction (`spectral_subtraction`)
- Frame-based STFT using `transforms.fft` inside each windowed frame (manual overlap-add)
- Profiles noise from a configurable silent segment; subtracts scaled magnitude spectrum
- Over-subtraction factor, Hann windowing, 75% overlap, ISTFT reconstruction via `ifft_real`

---

## Phase 2 — Signal Pipeline Integration

### `signal_core.py` — `EffectChainConfig`
A plain Python `@dataclass` with `__post_init__` validation:
- `echo_feedback < 1.0` (instability guard)
- `reverb_room_size` and `reverb_wet` in `[0, 1]`
- `eq_gains_db` must have exactly 5 values

### `main.py` — `run_effect_chain(sig, config)`
Orchestrates the 4-stage pipeline in fixed order:
```
Spectral Subtraction → Equalizer → Echo → Reverb
```
Only enabled stages execute. Lazily imports `effects` to avoid circular imports.

---

## Phase 3 — Streamlit UI (`app.py`)

New sidebar section **"🎛️ Effects Chain"** with three collapsible expanders:

| Widget | Control |
|---|---|
| Advanced Noise Removal checkbox | Enables spectral subtraction |
| Noise Profile Window slider | Sets profiling segment end (0.05–0.5 s) |
| Over-subtraction Factor slider | Controls aggressiveness (0.5–3.0) |
| 🎚️ Equalizer expander | 5 per-band gain sliders (−12 to +12 dB) |
| 🔁 Echo expander | Delay Time / Feedback / Wet Mix sliders |
| 🏛️ Reverb expander | Algorithmic / Convolution radio + Room Size / Damping / Wet |
| RIR file uploader | Appears when "Convolution (RIR)" is selected |

**Fourth audio player** added: "🎛️ Effects Output" alongside Noisy / Restored / Clean Reference.

---

## Phase 4 — Tests & Benchmark

### `test_all.py` — 18 new tests (5 sections)

| Section | Tests |
|---|---|
| Phase 5.1 EQ | flat 0 dB round-trip, +12 dB boost, −12 dB cut, LP energy reduction, wrong-gains ValueError |
| Phase 5.2 Echo | impulse peaks at 0 and M, exponential decay, feedback≥1 ValueError, wet=0 passthrough |
| Phase 5.3 Reverb | wet=0 passthrough, tail energy, room_size>1 ValueError, conv reverb length, fs mismatch |
| Phase 5.4 Spec. Sub. | length preserved, noise energy reduced, silence passthrough |
| Phase 5.5 Config | valid default, feedback ValueError, room_size ValueError, wrong EQ length ValueError |

### `benchmark.py` — `run_effects_benchmark` / `plot_effects_benchmark`
Times each effect at block sizes `[512, 1024, 4096, 8192, 16384]` samples; outputs `report/benchmark_effects.png`.

---

## Test Results

```
65/65 tests PASSED  ✓  All tests passed!
(47 original + 18 new Phase 5 tests)
```

---

## Constraint Compliance

| Constraint | Status |
|---|---|
| No `scipy.signal.*` in DSP | ✅ Biquad, echo, reverb, STFT all use NumPy only |
| No `numpy.fft.*` or `scipy.fft.*` | ✅ All FFTs route through `transforms.fft` |
| `scipy.io.wavfile` for I/O only | ✅ Only in `signal_core.py` `from_wav`/`to_wav` |
| Input to `transforms.fft` zero-padded to power of 2 | ✅ STFT frames use `next_power_of_two` |
| Streamlit audio via file paths (not bytes) | ✅ All 4 players use `_tmp_*.wav` file paths |
