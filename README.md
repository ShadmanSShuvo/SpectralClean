# SpectralClean — Audio Restoration & Analysis Studio

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spectralclean.streamlit.app)

**Live Demo:** [https://spectralclean.streamlit.app](https://spectralclean.streamlit.app)

A from-scratch DSP pipeline demonstrating DFT, FFT, convolution, spectral
denoising, and a full audio effects chain — **no `scipy.signal`, no `numpy.fft`**
in the core pipeline.


## Repository structure

```
signal_core.py   Phase 1.1–1.2  Signal class, EffectChainConfig, shift/scale/add/multiply
convolution.py   Phase 1.3      Manual convolution (direct summation)
transforms.py    Phase 2.1–2.3  Naive DFT · Cooley-Tukey FFT · IFFT
filters.py       Phase 3.3      Band-stop / low-pass / high-pass masks
effects.py       Phase 5        EQ · Echo/Delay · Reverb · Spectral Subtraction  ← NEW
metrics.py       Phase 4.1      SNR · PSNR · RMSE
benchmark.py     Phase 2.4      DFT vs FFT timing + effects overhead benchmarks
main.py          Phase 3–5      Full pipeline + run_effect_chain() orchestrator
app.py           Phase 1–5      Interactive Streamlit dashboard with effects controls
test_all.py      Phase 1–5      Unit-test suite (65 assertions)
requirements.txt               numpy · matplotlib · scipy (I/O only) · streamlit
report/                        Output figures and .wav files
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all unit tests first (65 assertions — 47 original + 18 new effects tests)
python test_all.py

# Run the interactive Streamlit dashboard
streamlit run app.py

# Run the terminal-based pipeline → generates report/*.wav + report/*.png
python main.py

# Standalone DFT vs FFT benchmark plot only
python benchmark.py

# To stop the app in a terminal, Ctrl+C the background task, or:
pkill -f "streamlit run app.py"
```

## Constraints honoured

| Module           | Allowed                            | Forbidden                    |
|------------------|------------------------------------|------------------------------|
| `signal_core.py` | NumPy, Matplotlib                  | scipy.signal, numpy.fft      |
| `convolution.py` | NumPy                              | np.convolve, scipy.signal    |
| `transforms.py`  | NumPy (math only)                  | numpy.fft, scipy.fft         |
| `filters.py`     | NumPy + transforms.py              | scipy.signal                 |
| `effects.py`     | NumPy + transforms.py              | scipy.signal, scipy.fft      |
| `metrics.py`     | NumPy                              | —                            |
| `benchmark.py`   | NumPy + transforms.py              | —                            |
| `main.py`        | All project modules                | —                            |
| `app.py`         | All project modules + Streamlit    | —                            |
| `test_all.py`    | numpy.fft **for validation only**  | —                            |

`scipy.io.wavfile` is used **only** in `signal_core.from_wav` / `to_wav` for
binary I/O. No signal processing functions from scipy are used anywhere.

## Key deliverables

### 4-Panel Denoising Figure (`report/four_panel.png`)

| Panel | Content |
|-------|---------|
| ① | Noisy waveform (time domain) |
| ② | Noisy spectrum — hum spikes clearly visible |
| ③ | Cleaned spectrum — notches applied |
| ④ | Restored waveform vs ground truth overlay |

### Benchmark Figure (`report/benchmark_dft_vs_fft.png`)

Log-log plot of DFT O(N²) vs FFT O(N log N) execution times from N=2⁴ to N=2¹⁴,
with theoretical reference curves and measured speedup annotation.

### Effects Benchmark Figure (`report/benchmark_effects.png`)  ← NEW

Processing overhead vs block size for each v2 effect (EQ, Echo, Reverb,
Spectral Subtraction) — manual NumPy implementations, no scipy.signal.

## v2 Audio Effects (`effects.py`)

All effects are implemented from scratch using NumPy only — no `scipy.signal.*`.

### Equalizer (IIR Biquad)
```
y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]
```
Biquad coefficient sets: `lowpass`, `highpass`, `peaking`, `lowshelf`, `highshelf`.
`Equalizer.graphic_eq(gains_db, fs)` maps 5 gain values to bands `[60, 230, 910, 4000, 14000]` Hz.

### Echo / Delay
```
y[n] = x[n] + g · y[n − M]      (feedback comb filter)
```
`M = delay_ms / 1000 · fs` samples; `g < 1.0` enforced to prevent instability.

### Algorithmic Reverb (Schroeder / Freeverb design)
4 parallel feedback comb filters (with 1-pole low-pass damping) fed into 2 series
all-pass diffuser filters. Configurable `room_size`, `damping`, and `wet` mix.

### Convolution Reverb
Convolves input with a custom Room Impulse Response (RIR) `.wav` file via the
existing `convolve_fast` implementation. Upload RIRs through the Streamlit UI.

### Spectral Subtraction (Advanced Noise Removal)
Manual STFT using `transforms.fft` frame-by-frame with Hann windowing and 75% overlap.
Profiles noise from a silent segment, subtracts the magnitude spectrum, reconstructs
via overlap-add ISTFT using `transforms.ifft_real`.

### Effect Chain
```
Spectral Subtraction → Equalizer → Echo → Reverb
```
Configured via `EffectChainConfig` dataclass in `signal_core.py` and orchestrated by
`run_effect_chain(sig, config)` in `main.py`.

## Mathematical foundations

### DFT (Phase 2.1)
```
X[k] = Σ_{n=0}^{N-1}  x[n] · e^{-j2πkn/N}
```
Implemented as a vectorised N×N matrix multiplication.

### Cooley-Tukey FFT (Phase 2.2)
```
X[k]       = E[k] + W_N^k · O[k]
X[k + N/2] = E[k] − W_N^k · O[k]
```
Even/odd recursive split; base case N=1. Requires N = 2^p (zero-padded).

### IFFT (Phase 2.3) — Conjugate trick
```
x[n] = (1/N) · conj( FFT( conj(X) ) )
```

### Denoising pipeline (Phase 3)
1. Zero-pad signal to next power of 2
2. FFT → X[k]
3. Build mask M[k] ∈ {0,1}: zero at hum bins, ones elsewhere
4. X_filtered[k] = M[k] · X[k]
5. IFFT → reconstructed signal, trim to original N

## SNR improvement (example run)

```
SNR before:    +2.49 dB   (50 Hz hum ×4 harmonics + σ=0.05 white noise)
SNR after:    +12.81 dB   (band-stop notching + 3.4 kHz low-pass)
Improvement:  +10.31 dB

PSNR before:  +11.87 dB   → PSNR after: +22.18 dB
RMSE before:   0.2168     → RMSE after:  0.0661
```

## Benchmark results (measured)

| N       | FFT (ms) | DFT (ms) | Speedup |
|---------|----------|----------|---------|
| 16      | 0.06     | 0.01     | 0.2×    |
| 256     | 0.93     | 0.94     | 1.0×    |
| 512     | 1.81     | 3.80     | 2.1×    |
| 1024    | 3.70     | 15.16    | 4.1×    |
| 2048    | 7.31     | 59.91    | **8.2×**|
| 4096+   | 14.76+   | —        | DFT too slow |

Crossover: FFT overtakes DFT around N=256; at N=2048 the gap is already 8×,
consistent with N log₂N / N² = log₂(2048)/2048 ≈ 1/186 theoretical ratio.
