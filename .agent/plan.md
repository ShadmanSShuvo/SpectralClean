# SpectralClean: Audio Restoration & Analysis Studio — Implementation Plan

## Phase 0 — Setup & Scope Lock
- Environment: Python, NumPy, Matplotlib, `scipy.io.wavfile` only for reading/writing `.wav` (I/O, not math) — no `scipy.signal`, no `numpy.fft`
- Repo structure:
  ```
  /signal_core.py      # Signal class, shift/scale/add
  /convolution.py       # manual convolution
  /transforms.py        # DFT, FFT, IFFT (scratch)
  /filters.py            # masking, band-stop/low-pass/high-pass
  /benchmark.py          # DFT vs FFT timing
  /metrics.py             # SNR calc
  /main.py                # pipeline orchestration
  /app.py                 # Interactive Streamlit dashboard GUI
  /report/                 # plots + writeup + temporary media
  ```
- Decide test audio: record/find a clean speech `.wav`, plan synthetic noise injection (tone hum + Gaussian white noise)

## Phase 1 — Signal & LTI Foundations
**Goal:** correctness of primitives everything else depends on.

- **1.1 Signal representation**: class/struct holding samples array, `fs` (sampling rate), time vector `t = n/fs`. Include a `.plot()` method (time-domain).
- **1.2 Elementary ops**: shift (index offset, zero-padded edges), amplitude scale, time scale, add/multiply two signals (handle length mismatch by zero-padding).
- **1.3 Manual convolution**: implement the direct summation formula (not `np.convolve`). Validate against a simple moving-average kernel on a spike-input test signal — confirm smoothing behavior visually and numerically.
- **Deliverable**: unit-test script showing shift/scale/add/convolve all match hand-calculated expected outputs on toy signals.

## Phase 2 — Spectral Transformation Engine
**Goal:** the mathematical core, the part graders scrutinize hardest.

- **2.1 Naive DFT**: implement $X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$ directly (nested loop or vectorized matrix form — vectorized version doubles as an optimization talking point).
- **2.2 Radix-2 Cooley-Tukey FFT**: recursive divide-and-conquer, even/odd split. Note constraint: input length must be padded to next power of 2 — implement zero-padding helper.
- **2.3 Inverse FFT (IFFT)**: derive from FFT (conjugate trick or direct inverse formula) — needed later for reconstruction.
- **2.4 Benchmark**: loop over increasing N (e.g., 2^4 to 2^14), time naive DFT vs FFT, plot execution time vs N on log-scale y-axis. This is a standalone, high-value deliverable — treat it as its own mini-report figure.
- **Validation**: cross-check scratch DFT/FFT output against `numpy.fft.fft` **only** for correctness testing, never in the shipped pipeline.

## Phase 3 — Denoising Application (core project)
**Goal:** apply Phase 1–2 machinery to a real, demonstrable problem.

- **3.1 Noise injection**: load clean `.wav`, add (a) a constant-frequency sine hum, (b) Gaussian white noise. Keep both a "ground truth clean" and "corrupted" version for later comparison.
- **3.2 Spectral analysis**: run scratch FFT on corrupted signal, plot magnitude spectrum — visually identify noise as narrow spikes (hum) vs raised noise floor (white noise).
- **3.3 Filter design**: 
  - Frequency-domain masking: zero out bins at/near the hum frequency (band-stop)
  - Optionally implement a convolution-based FIR low-pass as an alternate method, to show both approaches
- **3.4 Reconstruction**: apply IFFT to filtered spectrum, write result to a new `.wav`, listen/compare informally.

## Phase 4 — Verification & Metrics
**Goal:** prove the system works quantitatively, not just "it sounds better."

- **4.1 SNR calculation**: compute SNR before and after filtering (signal power vs residual noise power estimate); report the improvement in dB.
- **4.2 Final 4-panel figure**: noisy waveform, noisy spectrum (noise spikes visible), cleaned spectrum, restored waveform — this is the centerpiece figure for the report.
- **4.3 Edge case checks**: silence segments, clipping, very short signals, filter over-aggressiveness (audible artifacts).

## Phase 4.5 — Streamlit Interactive GUI
**Goal:** design a real-time reactive user interface to visualize and tune the DSP parameters.

- **4.5.1 Control Sidebar**: Sliders for Hum frequency (Hz), Hum amplitude, Gaussian white noise standard deviation ($\sigma$), low-pass cutoff frequency, notch bandwidth, and suppressed harmonics.
- **4.5.2 Audio Player Verification**: Write separate WAV files to disk (`_tmp_noisy.wav`, `_tmp_denoised.wav`, `_tmp_clean.wav`) rather than passing raw in-memory bytes, to prevent Streamlit's caching engine from serving the same media blob.
- **4.5.3 Real-time Plots**: Embed custom Matplotlib waveform, spectral, 4-panel, and complexity benchmark figures.

## Phase 5 — Report & Polish
- Write up: theory recap (DFT/FFT math, filter design rationale), methodology, benchmark graph + discussion, SNR results, 4-panel figure, limitations, Streamlit UI documentation
- Code cleanup: docstrings, consistent naming, remove debug prints, final run-through end-to-end
- Prepare for project update checkpoint and any viva — be ready to explain *why* FFT is faster (recursive halving) and *why* masking works (linearity of Fourier transform)

---

**Agent/execution notes:**
- Each phase is independently testable before moving on — don't let Phase 3 start until Phase 1–2 unit tests pass cleanly.
- The benchmark (2.4) and the 4-panel figure (4.2) are the two highest-leverage deliverables for grading — prioritize polish there if time is short.
- Ensure audio players are distinct in `app.py` by outputting to temp files.


