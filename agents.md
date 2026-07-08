# agents.md — Agent Workspace Constraints & Context

## Operational Mandate

This file provides system constraints, verification checkpoints, and behavioral guidelines for any AI agent or compiler executing tasks within the `SpectralClean` workspace.

## 🚨 Critical Code Constraints

1. **NO Mathematical Library Wrappers:**
   - **Allowed:** NumPy (`numpy`), Matplotlib (`matplotlib`), Standard Library.
   - **Allowed (I/O ONLY):** `scipy.io.wavfile` inside `signal_core.py` to read/write `.wav` formats.
   - **Strictly Forbidden:** `scipy.signal.*`, `numpy.fft.*`, `scipy.fft.*`, or any external library that implements Fourier transforms, windowing, filtering, or convolution.
   - *Rationale:* Graders inspect the codebase for manual implementations of these algorithms.

2. **FFT Constraints:**
   - Input to `transforms.fft()` must be zero-padded to a power of 2. Use `transforms.zero_pad_to_power2()` before calling the forward FFT.

3. **Streamlit Audio Handling:**
   - Never stream audio files using raw bytes or in-memory byte buffers via `st.audio(bytes)`. This triggers Streamlit's media file manager deduplication bugs.
   - Always write output tracks to the filesystem (`report/_tmp_noisy.wav`, etc.) and supply the file paths to the widgets.

## Verification Checkpoints

Before marked complete, a task must satisfy:

1. **Test Suite Execution:**
   - All 47 tests inside `test_all.py` must pass with zero failures.
   - Execute: `python test_all.py`
2. **Grader Verification:**
   - `main.py` must execute without error, outputting the metrics report and generating files in `report/`.
3. **No Lint Errors:**
   - Ensure no syntax warnings or runtime exceptions in `app.py`.
