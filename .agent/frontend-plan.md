## Streamlit Frontend Implementation Plan


### Phase 1 — Interactive Orchestration (`app.py`)

Create a new file named `app.py` at the root of your workspace. This file will serve as the graphical entry point, importing your verified logic directly from `signal_core`, `filters`, and `transforms`.

* **File Uploader:** Add an `st.file_uploader` widget restricted to `.wav` files. If no file is uploaded, fall back to your excellent `make_clean_signal()` generator so the app is immediately interactive on load.
* **Audio I/O Handling:** Use a simple helper to convert the uploaded binary stream into your custom `Signal` container using `scipy.io.wavfile`.

### Phase 2 — The Control Sidebar (Hyperparameter Tuning)

Give the user immediate control over the DSP variables using interactive sliders. Wrap these in an `st.sidebar` layout:

* **Noise Injection Controls:**
  * A checkbox to "Inject Synthetic Noise".
  * Sliders for Hum Frequency (40 Hz to 100 Hz), Hum Amplitude, and White Noise standard deviation ($\sigma$).
* **Filter Controls:**
  * A slider for the **Low-pass Cutoff Frequency** (1000 Hz to 4000 Hz).
  * A slider for the **Notch Filter Width** (bandwidth in Hz).
  * A number input for the number of hum harmonics to suppress.

### Phase 3 — Real-Time Signal Processing Loop

Inside the main app execution block, run your pipeline dynamically based on the slider configurations:

1. Capture the clean or uploaded signal.
2. If the noise checkbox is active, pass it through `inject_noise`.
3. Execute your custom `denoise_combined` function natively using the exact slider parameters chosen by the user.
4. Calculate the real-time SNR, PSNR, and RMSE metrics on the fly using `snr_improvement`.

### Phase 4 — Dynamic UI Rendering & Audio Verification

Streamlit handles reactive layout changes automatically. When a slider moves, the entire script reruns seamlessly.

* **Metric Callouts:** Display the quantitative performance at the very top using columns:
```python
col1, col2, col3, col4 = st.columns(4)
col1.metric("SNR Before", f"{metrics['snr_before_db']:+.2f} dB")
col2.metric("SNR After", f"{metrics['snr_after_db']:+.2f} dB", delta=f"{metrics['improvement_db']:+.2f} dB")
col3.metric("Final PSNR", f"{metrics['psnr_after_db']:.2f} dB")
col4.metric("RMSE Reduction", f"{metrics['rmse_after']:.4f}", delta=f"{metrics['rmse_after'] - metrics['rmse_before']:.4f}", delta_color="inverse")
```

* **File-Based Audio Players:**
  Instead of passing in-memory byte streams directly (which Streamlit's media server can aggressively cache and deduplicate, causing multiple players to play the same audio file), write the audio signals to temporary files on disk:
  ```python
  noisy_path    = "report/_tmp_noisy.wav"
  denoised_path = "report/_tmp_denoised.wav"
  clean_path    = "report/_tmp_clean.wav"
  
  to_wav(noisy_sig, noisy_path)
  to_wav(denoised_sig, denoised_path)
  to_wav(clean_sig, clean_path)
  ```
  Then display three audio players side-by-side:
  - **Noisy Audio** (`_tmp_noisy.wav`)
  - **Restored Audio** (`_tmp_denoised.wav`)
  - **Clean Reference** (`_tmp_clean.wav`)

* **Plot Embedding:** Pass your custom Matplotlib figures straight to `st.pyplot(fig)`. You can embed your 4-panel analysis plot or your log-log time complexity benchmark graph right into the dashboard layout.

---

## Updated Repository Tree

If you build this, your repository layout will look clean and structured:

```text
├── signal_core.py
├── convolution.py
├── transforms.py
├── filters.py
├── metrics.py
├── benchmark.py
├── main.py        <-- Stays as your terminal-based grader entry point
├── app.py         <-- NEW: Your interactive Streamlit dashboard interface
└── test_all.py    
```