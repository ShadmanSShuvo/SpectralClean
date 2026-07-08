# MEMORY.md — Project Memory and Log of Key Resolutions

## Project Summary
SpectralClean is an audio restoration and analysis application. Its core signal processing engine (Fourier transforms, convolution, frequency filtering, and metrics) is written completely from scratch using standard Python and NumPy, without relying on `scipy.signal` or `numpy.fft`.

## Implementation Chronicle
- **Phase 1 — Foundations:** Implemented `Signal` representation, elementary operations, and direct summation linear convolution. Verified with 1D impulses and moving average kernels.
- **Phase 2 — Transforms:** Implemented O(N²) Matrix DFT, Cooley-Tukey Radix-2 FFT, and IFFT via the conjugate trick.
- **Phase 3 — Denoising:** Created frequency-domain masking routines for low-pass and notch/band-stop filtering.
- **Phase 4 — Verification:** Added SNR, PSNR, and RMSE calculation modules.
- **Phase 4.5 — Frontend:** Designed a reactive Streamlit dashboard (`app.py`) allowing real-time parameter tuning.

## Key Issues Resolved

### 1. Streamlit Audio Player Conflation (Caching Bug)
- **Problem:** When trying to play the **Noisy Audio** and **Restored Audio** using raw bytes (`st.audio(bytes_data)`), both widgets played the exact same track. 
- **Investigation:** Streamlit hashes in-memory byte arrays to register them as static media resources. Due to similar sizing and quick updates, Streamlit's media file manager deduplicated the resources and served the first registered track to both players.
- **Resolution:** Modified the dashboard to write distinct WAV files to disk (`report/_tmp_noisy.wav`, `report/_tmp_denoised.wav`, and `report/_tmp_clean.wav`) using the core `to_wav()` function. Passing the distinct file paths to `st.audio(filepath)` completely bypassed Streamlit's media caching issue, allowing clean playback for all players.

### 2. Peak-Normalisation Volume Masking
- **Problem:** The restored audio did not sound perceptually quieter or less noisy than the corrupted signal.
- **Investigation:** The `wav_bytes_from_signal()` helper function peak-normalized both signals to `[-1, 1]` before encoding. This scaling erased the volume reduction (noise power suppression), making the denoised signal sound just as loud as the noisy one.
- **Resolution:** Added a `normalise` parameter defaulting to `False` in `wav_bytes_from_signal()`. Raw values are clipped to `[-1, 1]` without scaling, preserving the physical and audible difference in signal power.

### 3. DFT vs. FFT Crossover Metrics
- Naive DFT becomes bottlenecked above N = 2048.
- Measured crossover occurs at N = 256.
- At N = 2048, FFT is **8.2x faster** than DFT. Naive DFT is disabled for sizes greater than N = 2^11 in the UI and CLI benchmarks to prevent system hangs.
