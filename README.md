# SpectralClean — Audio Restoration & Analysis Studio

A from-scratch DSP pipeline demonstrating DFT, FFT, convolution, and
spectral denoising — **no `scipy.signal`, no `numpy.fft`** in the core pipeline.

## Repository structure

```
signal_core.py   Phase 1.1–1.2  Signal class, shift/scale/add/multiply
convolution.py   Phase 1.3      Manual convolution (direct summation)
transforms.py    Phase 2.1–2.3  Naive DFT · Cooley-Tukey FFT · IFFT
filters.py       Phase 3.3      Band-stop / low-pass / high-pass masks
metrics.py       Phase 4.1      SNR · PSNR · RMSE
benchmark.py     Phase 2.4      DFT vs FFT timing benchmark
main.py          Phase 3–4      Full pipeline (noise inject → denoise → figures)
test_all.py      Phase 1–4      Unit-test suite
requirements.txt               numpy · matplotlib · scipy (I/O only)
report/                        Output figures and .wav files
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run all unit tests first (47 assertions)
python test_all.py

# Run the full pipeline → generates report/*.wav + report/*.png
python main.py

# Standalone DFT vs FFT benchmark plot only
python benchmark.py
```

## Constraints honoured

| Module           | Allowed                       | Forbidden                    |
|------------------|-------------------------------|------------------------------|
| `signal_core.py` | NumPy, Matplotlib             | scipy.signal, numpy.fft      |
| `convolution.py` | NumPy                         | np.convolve, scipy.signal    |
| `transforms.py`  | NumPy (math only)             | numpy.fft, scipy.fft         |
| `filters.py`     | NumPy + transforms.py         | scipy.signal                 |
| `metrics.py`     | NumPy                         | —                            |
| `benchmark.py`   | NumPy + transforms.py         | —                            |
| `main.py`        | All project modules           | —                            |
| `test_all.py`    | numpy.fft **for validation only** | —                        |

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
