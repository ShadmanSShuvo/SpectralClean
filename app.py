"""
app.py — SpectralClean Interactive Dashboard
Streamlit frontend for the audio restoration pipeline.

Run:
    streamlit run app.py

All DSP logic is imported from the project modules — no scipy.signal.
scipy.io.wavfile is used only for binary I/O of uploaded files.
"""

import io
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")                 # headless backend for Streamlit
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import streamlit as st

# ── Project modules ───────────────────────────────────────────────────
from signal_core import Signal, to_wav, EffectChainConfig
from filters import denoise_combined
from transforms import fft, zero_pad_to_power2, fft_frequencies, fft_magnitude
from metrics import snr_improvement
from main import run_effect_chain


# ══════════════════════════════════════════════════════════════════════
#  Page config
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="SpectralClean — Audio Restoration Studio",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS (dark glassmorphism theme) ─────────────────────────────
st.markdown("""
<style>
    /* ── Base ──────────────────────────────────────────────────────── */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
    }
    [data-testid="stSidebar"] {
        background: rgba(22, 27, 34, 0.95);
        border-right: 1px solid #30363D;
    }
    .main .block-container { padding-top: 1.5rem; }

    /* ── Metric cards ──────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: rgba(33, 38, 45, 0.8);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        backdrop-filter: blur(8px);
    }
    [data-testid="stMetricLabel"]  { color: #8B949E !important; font-size: 0.78rem; }
    [data-testid="stMetricValue"]  { color: #E6EDF3 !important; font-size: 1.5rem; }
    [data-testid="stMetricDelta"]  { font-size: 0.82rem; }

    /* ── Section headers ───────────────────────────────────────────── */
    .section-header {
        font-size: 0.72rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        color: #8B949E;
        text-transform: uppercase;
        margin: 1.4rem 0 0.4rem;
        padding-bottom: 0.3rem;
        border-bottom: 1px solid #21262D;
    }

    /* ── Hero banner ───────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(90deg,
            rgba(0,229,255,0.12) 0%,
            rgba(105,255,71,0.07) 50%,
            rgba(255,107,107,0.10) 100%);
        border: 1px solid #30363D;
        border-radius: 16px;
        padding: 1.2rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .hero h1 { color: #00E5FF; margin: 0 0 0.2rem; font-size: 1.7rem; }
    .hero p  { color: #8B949E; margin: 0; font-size: 0.9rem; }

    /* ── Divider ───────────────────────────────────────────────────── */
    hr { border-color: #21262D; }

    /* ── Audio player ──────────────────────────────────────────────── */
    audio { width: 100%; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════

def compute_spectrum(sig: Signal):
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    X = fft(x_padded)
    freqs = fft_frequencies(N, sig.fs)
    mag = fft_magnitude(X)
    return X, freqs, mag, N


@st.cache_data(show_spinner=False)
def make_clean_signal_cached(fs: float = 8000.0, duration: float = 3.0) -> tuple:
    """
    Deterministic synthetic speech signal (cache avoids recompute on every slider move).
    Returns (samples_list, fs) — tuple so it's hashable for Streamlit cache.
    """
    from main import make_clean_signal
    sig = make_clean_signal(fs, duration)
    return sig.samples.tolist(), sig.fs


def wav_bytes_from_signal(sig: Signal, normalise: bool = False) -> bytes:
    """
    Encode a Signal to in-memory WAV bytes for st.audio().

    Parameters
    ----------
    sig       : Signal to encode
    normalise : If True, peak-normalise to [-1, 1] before encoding.
                Leave False (default) so the actual amplitude difference
                between noisy and denoised signals is audible.
    """
    from scipy.io import wavfile
    buf = io.BytesIO()
    if normalise:
        peak = np.max(np.abs(sig.samples))
        data = sig.samples / (peak if peak > 0 else 1.0)
    else:
        data = sig.samples
    out = (np.clip(data, -1, 1) * 32767).astype(np.int16)
    wavfile.write(buf, int(sig.fs), out)
    return buf.getvalue()


def load_uploaded_wav(uploaded) -> Signal:
    """Convert a Streamlit UploadedFile → Signal. I/O only via scipy.io.wavfile."""
    from scipy.io import wavfile
    data, fs = wavfile.read(io.BytesIO(uploaded.read()))
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return Signal(data, float(fs))


def inject_noise_fn(
    clean: Signal,
    hum_freq: float,
    hum_amp: float,
    white_std: float,
    harmonics: int,
    seed: int = 0,
) -> Signal:
    """Noise injection (mirrors main.inject_noise, kept local to avoid import cycle)."""
    t = clean.t
    hum = np.zeros(clean.N)
    for h in range(1, harmonics + 1):
        hum += (hum_amp / h) * np.sin(2 * np.pi * hum_freq * h * t)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, white_std, clean.N)
    return Signal(clean.samples + hum + noise, clean.fs)


# ══════════════════════════════════════════════════════════════════════
#  Plotting helpers (dark theme, Matplotlib → st.pyplot)
# ══════════════════════════════════════════════════════════════════════

DARK_BG  = "#0D1117"
PANEL_BG = "#161B22"
TEXT     = "#C9D1D9"
GRID     = "#21262D"

def _ax_style(ax, title):
    ax.set_facecolor(PANEL_BG)
    ax.set_title(title, color="white", fontsize=10, pad=6, fontweight="bold")
    ax.tick_params(colors=TEXT, labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor("#30363D")
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def plot_waveform_pair(noisy: Signal, denoised: Signal, clean: Signal = None) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
    fig.patch.set_facecolor(DARK_BG)

    ax1.plot(noisy.t, noisy.samples, color="#FF6B6B", linewidth=0.6, alpha=0.9)
    _ax_style(ax1, "Noisy Waveform")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude")

    ax2.plot(denoised.t, denoised.samples, color="#00E5FF", linewidth=0.6, alpha=0.9)
    if clean is not None:
        ax2.plot(clean.t, clean.samples, color="white", linewidth=0.4,
                 alpha=0.35, linestyle="--", label="ground truth")
        ax2.legend(facecolor="#1C2128", labelcolor=TEXT, fontsize=7.5)
    _ax_style(ax2, "Restored Waveform")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Amplitude")

    plt.tight_layout()
    return fig


def plot_spectrum_pair(
    noisy: Signal,
    denoised: Signal,
    hum_freq: float,
    harmonics: int,
) -> plt.Figure:
    _, fn, mn, _ = compute_spectrum(noisy)
    _, fd, md, _ = compute_spectrum(denoised)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3.5))
    fig.patch.set_facecolor(DARK_BG)

    ax1.plot(fn, mn, color="#FFA726", linewidth=0.8, alpha=0.9)
    ax1.set_xlim(0, min(noisy.fs / 2, 1500))
    for h in range(1, harmonics + 1):
        fh = hum_freq * h
        if fh <= 1500:
            ax1.axvline(fh, color="#FF4081", linewidth=1.0, linestyle="--", alpha=0.7)
    _ax_style(ax1, "Noisy Spectrum  (hum spikes visible)")
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")

    ax2.plot(fd, md, color="#69FF47", linewidth=0.8, alpha=0.9)
    ax2.set_xlim(0, min(noisy.fs / 2, 1500))
    _ax_style(ax2, "Cleaned Spectrum  (notches applied)")
    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Magnitude")

    plt.tight_layout()
    return fig


def plot_four_panel(clean, noisy, denoised, hum_freq, harmonics) -> plt.Figure:
    """Full 4-panel figure (same as main.py) embedded inline."""
    _, fn, mn, _ = compute_spectrum(noisy)
    _, fd, md, _ = compute_spectrum(denoised)

    fig = plt.figure(figsize=(13, 7))
    fig.patch.set_facecolor(DARK_BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    ax1.plot(noisy.t, noisy.samples, color="#FF6B6B", linewidth=0.6, alpha=0.85)
    _ax_style(ax1, "① Noisy Waveform")
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Amplitude")

    ax2.plot(fn, mn, color="#FFA726", linewidth=0.7, alpha=0.9)
    ax2.set_xlim(0, 1000)
    for h in range(1, harmonics + 1):
        fh = hum_freq * h
        if fh <= 1000:
            ax2.axvline(fh, color="#FF4081", linewidth=1.0, linestyle="--", alpha=0.7)
    _ax_style(ax2, "② Noisy Spectrum  — hum spikes visible")
    ax2.set_xlabel("Frequency (Hz)"); ax2.set_ylabel("Magnitude")

    ax3.plot(fd, md, color="#69FF47", linewidth=0.7, alpha=0.9)
    ax3.set_xlim(0, 1000)
    _ax_style(ax3, "③ Cleaned Spectrum  — notches applied")
    ax3.set_xlabel("Frequency (Hz)"); ax3.set_ylabel("Magnitude")

    ax4.plot(denoised.t, denoised.samples, color="#00E5FF", linewidth=0.6, alpha=0.85)
    ax4.plot(clean.t, clean.samples, color="white", linewidth=0.4,
             alpha=0.35, linestyle="--", label="ground truth")
    _ax_style(ax4, "④ Restored Waveform")
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Amplitude")
    ax4.legend(facecolor="#1C2128", labelcolor=TEXT, fontsize=7.5)

    fig.suptitle("SpectralClean — Audio Restoration Analysis",
                 color="white", fontsize=12, y=0.97)
    plt.tight_layout()
    return fig


def plot_benchmark_inline() -> plt.Figure:
    """Inline benchmark — lighter version (DFT up to 2^10 to stay fast in UI)."""
    from benchmark import run_benchmark, plot_benchmark
    results = run_benchmark(n_min_exp=4, n_max_exp=12, dft_max_exp=10, repeats=2)
    return plot_benchmark(results)


# ══════════════════════════════════════════════════════════════════════
#  SIDEBAR — Controls
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🎙️ SpectralClean")
    st.markdown("<p style='color:#8B949E;font-size:0.82rem;margin-top:-0.5rem;'>"
                "Audio Restoration Studio</p>", unsafe_allow_html=True)
    st.divider()

    # ── Signal source ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Signal Source</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a .wav file", type=["wav"],
        help="Leave empty to use the built-in synthetic speech signal"
    )
    if uploaded_file is None:
        st.caption("ℹ️ Using synthetic speech signal (3 s, 8 kHz)")

    st.divider()

    # ── Noise injection ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Noise Injection</div>', unsafe_allow_html=True)
    inject_noise = st.checkbox("Inject Synthetic Noise", value=True)

    hum_freq = st.slider(
        "Hum Frequency (Hz)", min_value=40, max_value=100,
        value=50, step=5,
        disabled=not inject_noise,
        help="Fundamental mains-hum frequency. 50 Hz (EU) or 60 Hz (US)."
    )
    hum_amp = st.slider(
        "Hum Amplitude", min_value=0.05, max_value=0.5,
        value=0.25, step=0.05,
        disabled=not inject_noise,
        help="Amplitude of hum relative to signal."
    )
    white_std = st.slider(
        "White Noise σ", min_value=0.01, max_value=0.20,
        value=0.05, step=0.01,
        disabled=not inject_noise,
        help="Standard deviation of additive Gaussian noise."
    )
    hum_harmonics_noise = st.slider(
        "Injected Harmonics", min_value=1, max_value=6,
        value=4,
        disabled=not inject_noise,
        help="Number of hum harmonics to inject."
    )

    st.divider()

    # ── Filter controls ───────────────────────────────────────────────
    st.markdown('<div class="section-header">Filter Controls</div>', unsafe_allow_html=True)
    lp_cutoff = st.slider(
        "Low-pass Cutoff (Hz)", min_value=1000, max_value=4000,
        value=3400, step=100,
        help="Attenuates all content above this frequency."
    )
    notch_bw = st.slider(
        "Notch Bandwidth (Hz)", min_value=1, max_value=20,
        value=3, step=1,
        help="Width of each band-stop notch around each hum harmonic."
    )
    filter_harmonics = st.number_input(
        "Harmonics to Suppress", min_value=1, max_value=8,
        value=4,
        help="How many hum harmonics the filter will notch out."
    )

    st.divider()

    # ── Effects Chain ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">🎛️ Effects Chain</div>', unsafe_allow_html=True)

    # Advanced Noise Removal
    adv_noise_removal = st.checkbox(
        "Advanced Noise Removal (Spectral Subtraction)",
        value=False,
        help="STFT-based spectral subtraction — profiles noise from a quiet segment "
             "and subtracts it from all frames. More aggressive than mask filtering."
    )
    if adv_noise_removal:
        nr_end = st.slider(
            "Noise Profile Window (s)", min_value=0.05, max_value=0.5,
            value=0.2, step=0.05,
            help="Duration of the initial silent segment used to profile the noise."
        )
        nr_alpha = st.slider(
            "Over-subtraction Factor", min_value=0.5, max_value=3.0,
            value=1.0, step=0.1,
            help="Scale the subtracted noise profile. >1.0 is more aggressive."
        )
    else:
        nr_end = 0.2
        nr_alpha = 1.0

    st.markdown("")

    # Equalizer
    with st.expander("🎚️ Equalizer (5-Band)", expanded=False):
        eq_enabled = st.checkbox("Enable EQ", value=False, key="eq_on")
        st.caption("Band gains in dB (−12 to +12). 0 = flat.")
        eq_col1, eq_col2 = st.columns(2)
        with eq_col1:
            eq_60   = st.slider("60 Hz",   -12, 12, 0, disabled=not eq_enabled, key="eq60")
            eq_230  = st.slider("230 Hz",  -12, 12, 0, disabled=not eq_enabled, key="eq230")
            eq_910  = st.slider("910 Hz",  -12, 12, 0, disabled=not eq_enabled, key="eq910")
        with eq_col2:
            eq_4k   = st.slider("4 kHz",   -12, 12, 0, disabled=not eq_enabled, key="eq4k")
            eq_14k  = st.slider("14 kHz",  -12, 12, 0, disabled=not eq_enabled, key="eq14k")
        eq_gains = [float(eq_60), float(eq_230), float(eq_910), float(eq_4k), float(eq_14k)]

    # Echo / Delay
    with st.expander("🔁 Echo / Delay", expanded=False):
        echo_enabled = st.checkbox("Enable Echo", value=False, key="echo_on")
        echo_delay = st.slider(
            "Delay Time (ms)", 10, 800, 200, disabled=not echo_enabled, key="echo_delay"
        )
        echo_feedback = st.slider(
            "Feedback", 0.0, 0.9, 0.3, step=0.05, disabled=not echo_enabled, key="echo_fb",
            help="Echo decay gain. Must be < 1.0 to remain stable."
        )
        echo_wet = st.slider(
            "Wet Mix", 0.0, 1.0, 0.5, step=0.05, disabled=not echo_enabled, key="echo_wet"
        )

    # Reverb
    with st.expander("🏛️ Reverb / Room Acoustics", expanded=False):
        reverb_enabled = st.checkbox("Enable Reverb", value=False, key="rev_on")
        reverb_mode = st.radio(
            "Reverb Type",
            ["Algorithmic (Schroeder)", "Convolution (RIR)"],
            disabled=not reverb_enabled, key="rev_mode"
        )
        room_size = st.slider(
            "Room Size", 0.0, 1.0, 0.5, step=0.05,
            disabled=not reverb_enabled, key="room_size",
            help="Controls comb filter feedback. Larger = longer reverb tail."
        )
        reverb_damping = st.slider(
            "Damping", 0.0, 1.0, 0.5, step=0.05,
            disabled=(not reverb_enabled or reverb_mode != "Algorithmic (Schroeder)"),
            key="rev_damp",
            help="High-frequency absorption inside the reverb loop."
        )
        reverb_wet = st.slider(
            "Wet Mix", 0.0, 1.0, 0.33, step=0.05,
            disabled=not reverb_enabled, key="rev_wet"
        )
        rir_file = None
        if reverb_enabled and reverb_mode == "Convolution (RIR)":
            rir_file = st.file_uploader(
                "Upload RIR .wav file", type=["wav"],
                key="rir_upload",
                help="Room Impulse Response file. Your audio will be convolved with this RIR."
            )
            if rir_file is None:
                st.caption("ℹ️ No RIR uploaded — falling back to algorithmic reverb.")

    st.divider()
    st.markdown('<div class="section-header">Visualisation</div>', unsafe_allow_html=True)
    show_4panel = st.checkbox("Show 4-Panel Analysis Figure", value=False)
    show_benchmark = st.checkbox("Show DFT vs FFT Benchmark", value=False)


# ══════════════════════════════════════════════════════════════════════
#  MAIN — Hero banner
# ══════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="hero">
  <h1>🎙️ SpectralClean</h1>
  <p>Audio Restoration &amp; Analysis Studio — real-time spectral denoising
  built from scratch (Cooley-Tukey FFT · band-stop masking · IFFT reconstruction).
  Move any slider to instantly reprocess the signal.</p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════
#  PIPELINE — Runs on every Streamlit rerun
# ══════════════════════════════════════════════════════════════════════

with st.spinner("Processing signal…"):

    # Step 1 — Acquire clean signal
    if uploaded_file is not None:
        uploaded_file.seek(0)
        clean_sig = load_uploaded_wav(uploaded_file)
        source_label = f"Uploaded: `{uploaded_file.name}`"
    else:
        samples, fs = make_clean_signal_cached()
        clean_sig = Signal(np.array(samples), fs)
        source_label = "Synthetic speech signal (built-in)"

    # Step 2 — Noise injection
    if inject_noise:
        noisy_sig = inject_noise_fn(
            clean_sig,
            hum_freq=float(hum_freq),
            hum_amp=hum_amp,
            white_std=white_std,
            harmonics=hum_harmonics_noise,
        )
    else:
        noisy_sig = clean_sig   # pass-through (filter still applied)

    # Step 3 — Denoise (hum mask)
    denoised_sig = denoise_combined(
        noisy_sig,
        hum_freq_hz=float(hum_freq),
        hum_bandwidth_hz=float(notch_bw),
        harmonics=int(filter_harmonics),
        lowpass_cutoff_hz=float(lp_cutoff),
    )

    # Step 4 — Effects Chain
    # Build RIR signal if a file was uploaded
    rir_sig = None
    if reverb_enabled and reverb_mode == "Convolution (RIR)" and rir_file is not None:
        rir_file.seek(0)
        rir_sig = load_uploaded_wav(rir_file)

    fx_config = EffectChainConfig(
        noise_removal=adv_noise_removal,
        noise_end_s=float(nr_end),
        noise_over_subtraction=float(nr_alpha),
        eq_enabled=eq_enabled,
        eq_gains_db=eq_gains,
        echo_enabled=echo_enabled,
        echo_delay_ms=float(echo_delay),
        echo_feedback=float(echo_feedback),
        echo_wet=float(echo_wet),
        reverb_enabled=reverb_enabled,
        reverb_room_size=float(room_size),
        reverb_damping=float(reverb_damping),
        reverb_wet=float(reverb_wet),
        rir_signal=rir_sig,
    )
    effects_sig = run_effect_chain(denoised_sig, fx_config)

    # Step 5 — Metrics
    metrics = snr_improvement(clean_sig, noisy_sig, denoised_sig)


# ══════════════════════════════════════════════════════════════════════
#  METRICS ROW
# ══════════════════════════════════════════════════════════════════════

st.markdown("### 📊 Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric(
    "SNR Before",
    f"{metrics['snr_before_db']:+.2f} dB",
)
col2.metric(
    "SNR After",
    f"{metrics['snr_after_db']:+.2f} dB",
    delta=f"{metrics['improvement_db']:+.2f} dB",
)
col3.metric(
    "Final PSNR",
    f"{metrics['psnr_after_db']:.2f} dB",
    delta=f"{metrics['psnr_after_db'] - metrics['psnr_before_db']:+.2f} dB",
)
col4.metric(
    "RMSE Reduction",
    f"{metrics['rmse_after']:.4f}",
    delta=f"{metrics['rmse_after'] - metrics['rmse_before']:.4f}",
    delta_color="inverse",
)

st.divider()


# ══════════════════════════════════════════════════════════════════════
#  AUDIO PLAYERS
# ══════════════════════════════════════════════════════════════════════

st.markdown("### 🔊 Listen & Compare")
st.caption(f"Source: {source_label}")


# Streamlit's in-memory bytes media server can silently serve the same
# cached audio blob for multiple st.audio() calls in one rerun.
# Fix: write to distinct temp files on disk and pass file paths.
import os, hashlib

_tmp_dir = os.path.join(os.path.dirname(__file__), "report")
os.makedirs(_tmp_dir, exist_ok=True)

noisy_path    = os.path.join(_tmp_dir, "_tmp_noisy.wav")
denoised_path = os.path.join(_tmp_dir, "_tmp_denoised.wav")
clean_path    = os.path.join(_tmp_dir, "_tmp_clean.wav")
effects_path  = os.path.join(_tmp_dir, "_tmp_effects.wav")

to_wav(noisy_sig, noisy_path)
to_wav(denoised_sig, denoised_path)
to_wav(clean_sig, clean_path)
to_wav(effects_sig, effects_path)

ac1, ac2, ac3, ac4 = st.columns(4)

with ac1:
    st.markdown("**🔴 Noisy Audio**")
    st.audio(noisy_path, format="audio/wav")
with ac2:
    st.markdown("**🟢 Restored Audio**")
    st.audio(denoised_path, format="audio/wav")
with ac3:
    st.markdown("**🎛️ Effects Output**")
    st.audio(effects_path, format="audio/wav")
with ac4:
    st.markdown("**⚪ Clean Reference**")
    st.audio(clean_path, format="audio/wav")

st.divider()


# ══════════════════════════════════════════════════════════════════════
#  WAVEFORM & SPECTRUM PLOTS
# ══════════════════════════════════════════════════════════════════════

st.markdown("### 📈 Waveform Analysis")
fig_wave = plot_waveform_pair(noisy_sig, denoised_sig, clean_sig if not inject_noise or uploaded_file is None else None)
st.pyplot(fig_wave, use_container_width=True)
plt.close(fig_wave)

st.markdown("### 🌊 Spectral Analysis")
st.caption(
    f"Hum at {hum_freq} Hz and {filter_harmonics} harmonics "
    f"({hum_freq}–{hum_freq * int(filter_harmonics)} Hz) are notched. "
    f"Low-pass cutoff: {lp_cutoff} Hz."
)
fig_spec = plot_spectrum_pair(noisy_sig, denoised_sig, float(hum_freq), int(filter_harmonics))
st.pyplot(fig_spec, use_container_width=True)
plt.close(fig_spec)


# ══════════════════════════════════════════════════════════════════════
#  OPTIONAL: 4-Panel figure
# ══════════════════════════════════════════════════════════════════════

if show_4panel:
    st.divider()
    st.markdown("### 🖼️ Full 4-Panel Analysis")
    with st.spinner("Rendering 4-panel figure…"):
        fig4 = plot_four_panel(
            clean_sig, noisy_sig, denoised_sig,
            float(hum_freq), int(filter_harmonics)
        )
    st.pyplot(fig4, use_container_width=True)
    plt.close(fig4)


# ══════════════════════════════════════════════════════════════════════
#  OPTIONAL: Benchmark
# ══════════════════════════════════════════════════════════════════════

if show_benchmark:
    st.divider()
    st.markdown("### ⏱️ DFT vs FFT Benchmark")
    st.caption("Naive O(N²) DFT vs Cooley-Tukey O(N log N) FFT — both from scratch.")
    with st.spinner("Running benchmark (may take ~20 s) …"):
        fig_bench = plot_benchmark_inline()
    st.pyplot(fig_bench, use_container_width=True)
    plt.close(fig_bench)


# ══════════════════════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════════════════════

st.divider()
st.markdown("""
<p style='color:#484F58;font-size:0.78rem;text-align:center;'>
SpectralClean · No <code>scipy.signal</code> · No <code>numpy.fft</code> in the DSP core ·
Cooley-Tukey FFT + band-stop masking implemented from scratch
</p>
""", unsafe_allow_html=True)
