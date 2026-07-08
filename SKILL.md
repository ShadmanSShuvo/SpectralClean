---
name: spectral-clean-dsp-engine
description: Guides agents in maintaining and verifying the from-scratch DSP and Streamlit audio pipeline.
---

# SpectralClean DSP Engine Skill

Use this skill when modifying, maintaining, or documenting the mathematical core (DFT, FFT, IFFT, convolution, filters) or the interactive Streamlit user interface.

## Core API Reference

### 1. Signal Representation (`signal_core.py`)
- Instantiate a signal: `Signal(samples: np.ndarray, fs: float)`
- Key operations:
  - `sig.shift(k: int)` -> zero-padded shift
  - `sig.scale(alpha: float)` -> amplitude scaling
  - `sig.time_scale(factor: float)` -> nearest-neighbor resampling
  - `sig.add(other)` / `sig.multiply(other)` -> element-wise operations with length mismatch safety (zero-pads the shorter signal)

### 2. Manual Convolution (`convolution.py`)
- Linear convolution via direct summation: `convolve(x: Signal, h: Signal)`
- Fast sliding dot product: `convolve_fast(x: Signal, h: Signal)`
- Sinc low-pass FIR kernel generation: `fir_lowpass_kernel(cutoff_hz, fs, num_taps)`

### 3. Spectral Transform Engine (`transforms.py`)
- Naive O(N²) DFT via vectorized matrix multiplication: `dft(x: np.ndarray)`
- Cooley-Tukey Radix-2 FFT: `fft(x: np.ndarray)` (Requires length to be a power of 2)
- Inverse FFT via the conjugate trick: `ifft(X: np.ndarray)` -> returns complex; `ifft_real(X: np.ndarray)` -> returns real part

### 4. Denoising & Filtering (`filters.py`)
- Multi-notch hum removal: `denoise_hum(sig, hum_freq_hz, bandwidth_hz, harmonics)`
- Low-pass masking: `denoise_lowpass(sig, cutoff_hz)`
- Combined band-stop and low-pass filter: `denoise_combined(sig, hum_freq_hz, hum_bandwidth_hz, harmonics, lowpass_cutoff_hz)`

## Streamlit Development Guardrails

1. **Avoid In-Memory Audio Playback:** Streamlit's `st.audio(bytes)` path hashes the byte array. If two players receive similar or identical-sized byte streams, the browser or Streamlit caching can conflate them.
   - **Correct Pattern:** Write WAV data to disk using `to_wav(sig, filepath)` and pass the string filepath to `st.audio(filepath)`.
2. **Preserve Volumetric Differences:** When rendering audio players, do **not** peak-normalize the noisy and restored outputs to the same volume in the conversion step. Leave `normalise=False` in `wav_bytes_from_signal` so the user can hear the physical volume reduction in noise.
3. **Headless Matplotlib:** Ensure `matplotlib.use("Agg")` is called before importing `pyplot` to prevent GUI rendering threads from crashing the Streamlit execution loop.
