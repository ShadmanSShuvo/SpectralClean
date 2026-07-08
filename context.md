# context.md — Developer and Agent System Context

This file provides system context, engineering constraints, and operational details for the `SpectralClean` workspace.

## System Capabilities

- **Time-Domain Signal Primitives:** Custom `Signal` container managing samples, sample rate (`fs`), and time axis. Supports delay/advance shifting, amplitude scaling, nearest-neighbor scaling, addition, and multiplication with automatic zero-padding length alignment.
- **textbook Convolution:** Direct summation linear convolution ($O(N_x \times N_h)$) and sliding dot product vectorised matrix convolution.
- **Custom Transforms Engine:** Symmetrical vectorized Matrix DFT, recursive Cooley-Tukey Radix-2 FFT (requires power-of-2 zero-padded input), and IFFT via the conjugate trick.
- **Notch and Band-Stop Masking:** Symmetrical notch filtering in the frequency domain at the fundamental and harmonics of mains hum, combined with high-frequency attenuation via low-pass filter.
- **Interactive UI Frontend:** Responsive Streamlit app (`app.py`) allowing manual parameter sweeps for hum frequency, hum amplitude, white noise std dev, and filtering cutoff frequencies with real-time audio comparison.

## Workspace constraints

- No `scipy.signal` or `numpy.fft` packages are permitted in the DSP modules (`signal_core.py`, `convolution.py`, `transforms.py`, `filters.py`).
- Audio files in Streamlit must be streamed using disk-bound paths (`report/_tmp_noisy.wav`, `report/_tmp_denoised.wav`, `report/_tmp_clean.wav`) rather than raw byte blobs to prevent browser/Streamlit caching bugs.

## Core Operations Checklist

- **Test Suite:** `python test_all.py` runs all 47 tests. All must pass.
- **CLI Orchestrator:** `python main.py` runs the full restoration pipeline.
- **Benchmark:** `python benchmark.py` compares O(N²) DFT vs. O(N log N) FFT.
