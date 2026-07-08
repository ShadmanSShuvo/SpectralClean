# CLAUDE.md — Developer Reference Guide

## Build & Dependencies Setup

The codebase runs in a Python virtual environment. It does not use `scipy.signal` or `numpy.fft` for any processing logic.

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Commands

- **Unit Tests:** Runs the full DSP validation suite.
  ```bash
  python test_all.py
  ```
- **Streamlit Interactive UI:** Launches the reactive web app dashboard.
  ```bash
  streamlit run app.py
  ```
- **CLI Pipeline Orchestration:** Runs the deterministic restoration script and generates figures.
  ```bash
  python main.py
  ```
- **Performance Benchmark:** Computes and plots naive DFT vs. Cooley-Tukey FFT execution times.
  ```bash
  python benchmark.py
  ```

## Coding Style & Architecture Guidelines

- **Libraries Constraint:** Never use `scipy.signal`, `numpy.fft`, `scipy.fft`, or other library-wrapped Fourier/convolution routines in the core processing files (`signal_core.py`, `convolution.py`, `transforms.py`, `filters.py`).
- **I/O Exception:** `scipy.io.wavfile` is explicitly allowed *only* for reading/writing binary `.wav` files.
- **Precision:** Use `np.float64` for sample processing and `np.complex128` for spectral calculations.
- **Audio Streaming:** When displaying audio in Streamlit, always stream using temporary disk-bound `.wav` files (`report/_tmp_noisy.wav`, etc.) rather than in-memory byte buffers to bypass browser and Streamlit-level cache duplication bugs.
- **Code Documentation:** Preserve docstrings describing mathematical derivations (matrix DFT, Cooley-Tukey butterfly, conjugate IFFT).
