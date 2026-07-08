"""
main.py — Pipeline orchestration
SpectralClean: Audio Restoration & Analysis Studio

Phases executed:
  1. Generate / load synthetic clean speech (chirp + harmonics)
  2. Inject noise: 50 Hz hum (+ harmonics) + Gaussian white noise
  3. Spectral analysis via scratch FFT
  4. Denoising: band-stop masking + optional low-pass
  5. IFFT reconstruction → output .wav
  6. SNR metrics before/after
  7. 4-panel figure (Phase 4.2) + benchmark plot (Phase 2.4)

Run: python main.py
"""

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ── Local modules ──────────────────────────────────────────────────────
from signal_core import Signal, sine_wave, white_noise, to_wav
from transforms import fft, zero_pad_to_power2, fft_frequencies, fft_magnitude
from filters import denoise_combined, apply_spectral_filter, band_stop_mask
from metrics import snr_improvement, print_metrics_report


# ══════════════════════════════════════════════════════════════════════
#  CONSTANTS & CONFIGURATION
# ══════════════════════════════════════════════════════════════════════
FS          = 8000          # sampling rate (Hz) — telephone quality
DURATION    = 3.0           # seconds
HUM_FREQ    = 50.0          # Hz (mains hum; 60.0 for US standard)
HUM_AMP     = 0.25          # hum amplitude relative to signal
WHITE_STD   = 0.05          # Gaussian noise std dev
HARMONICS   = 4             # number of hum harmonics to suppress
HUM_BW      = 3.0           # notch width per harmonic (Hz)
LP_CUTOFF   = 3400.0        # low-pass cutoff (Hz) — remove high-freq noise

REPORT_DIR  = Path("report")
REPORT_DIR.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
#  STEP 1 — Synthetic clean speech signal
# ══════════════════════════════════════════════════════════════════════

def make_clean_signal(fs: float = FS, duration: float = DURATION) -> Signal:
    """
    Synthesise a speech-like signal: sum of harmonic partials with a
    slowly evolving pitch contour (frequency sweep + vibrato), mimicking
    voiced speech formant structure.

    No audio file required — fully reproducible and deterministic.
    """
    N = int(fs * duration)
    t = np.arange(N) / fs

    # Fundamental + harmonics (speech formant approximation)
    f0_contour = 150 + 30 * np.sin(2 * np.pi * 0.5 * t)   # 150±30 Hz pitch

    signal = np.zeros(N)
    harmonic_amps = [1.0, 0.6, 0.35, 0.2, 0.12, 0.07, 0.04]
    for i, amp in enumerate(harmonic_amps, 1):
        phase_accum = 2 * np.pi * np.cumsum(f0_contour * i) / fs
        signal += amp * np.sin(phase_accum)

    # Apply Hann-windowed amplitude envelope (speech-like on/off ramps)
    envelope = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / N))
    signal *= envelope

    # Normalise to [-0.85, 0.85]
    peak = np.max(np.abs(signal))
    if peak > 0:
        signal = signal / peak * 0.85

    return Signal(signal, fs)


# ══════════════════════════════════════════════════════════════════════
#  STEP 2 — Noise injection
# ══════════════════════════════════════════════════════════════════════

def inject_noise(
    clean: Signal,
    hum_freq: float = HUM_FREQ,
    hum_amp: float = HUM_AMP,
    white_std: float = WHITE_STD,
    harmonics: int = HARMONICS,
    seed: int = 0,
) -> Signal:
    """
    Add (a) mains-hum tone + harmonics and (b) Gaussian white noise.

    Returns the corrupted Signal.
    """
    t = clean.t
    hum = np.zeros(clean.N)
    for h in range(1, harmonics + 1):
        amp_h = hum_amp / h  # harmonics decay with 1/h
        hum += amp_h * np.sin(2 * np.pi * hum_freq * h * t)

    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, white_std, clean.N)

    corrupted = clean.samples + hum + noise
    return Signal(corrupted, clean.fs)


# ══════════════════════════════════════════════════════════════════════
#  STEP 3 — Spectral analysis helper
# ══════════════════════════════════════════════════════════════════════

def compute_spectrum(sig: Signal):
    """Returns (X, freqs, mag) using scratch FFT."""
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    X = fft(x_padded)
    freqs = fft_frequencies(N, sig.fs)
    mag = fft_magnitude(X)
    return X, freqs, mag, N


# ══════════════════════════════════════════════════════════════════════
#  STEP 7a — 4-Panel Figure (Phase 4.2)
# ══════════════════════════════════════════════════════════════════════

def plot_four_panel(
    clean: Signal,
    noisy: Signal,
    denoised: Signal,
    save_path: str = "report/four_panel.png",
) -> plt.Figure:
    """
    Generate the centrepiece 4-panel figure:

    Panel 1 (top-left)  : Noisy waveform — time domain
    Panel 2 (top-right) : Noisy spectrum — frequency domain (hum spikes visible)
    Panel 3 (bottom-left): Cleaned spectrum — hum spikes removed
    Panel 4 (bottom-right): Restored waveform — time domain
    """
    _, freqs_n, mag_n, _ = compute_spectrum(noisy)
    _, freqs_d, mag_d, _ = compute_spectrum(denoised)

    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#0D1117")

    gs = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.32,
                           left=0.07, right=0.97, top=0.91, bottom=0.09)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    BG      = "#161B22"
    TEXT    = "#C9D1D9"
    GRID    = "#21262D"
    NOISY_C = "#FF6B6B"
    CLEAN_C = "#00E5FF"
    SPEC_N  = "#FFA726"
    SPEC_D  = "#69FF47"

    def _style(ax, title):
        ax.set_facecolor(BG)
        ax.set_title(title, color="white", fontsize=10.5, pad=6, fontweight="bold")
        ax.tick_params(colors=TEXT, labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # ── Panel 1: Noisy waveform ──────────────────────────────────────
    ax1.plot(noisy.t, noisy.samples, color=NOISY_C, linewidth=0.6, alpha=0.85)
    _style(ax1, "① Noisy Waveform  (time domain)")
    ax1.set_xlabel("Time (s)", color=TEXT, fontsize=8)
    ax1.set_ylabel("Amplitude", color=TEXT, fontsize=8)

    # ── Panel 2: Noisy spectrum ──────────────────────────────────────
    # Show up to Nyquist but zoom into lower frequencies to see hum spikes
    nyq = noisy.fs / 2
    ax2.plot(freqs_n, mag_n, color=SPEC_N, linewidth=0.7, alpha=0.9)
    ax2.set_xlim(0, 1000)  # zoom to 0-1 kHz to expose hum structure
    _style(ax2, "② Noisy Spectrum  — hum spikes visible (0–1 kHz)")
    ax2.set_xlabel("Frequency (Hz)", color=TEXT, fontsize=8)
    ax2.set_ylabel("Magnitude", color=TEXT, fontsize=8)

    # Mark hum harmonics
    for h in range(1, HARMONICS + 1):
        fh = HUM_FREQ * h
        if fh <= 1000:
            ax2.axvline(fh, color="#FF4081", linewidth=1.0, linestyle="--",
                        alpha=0.7, label=f"{int(fh)} Hz" if h == 1 else None)
    ax2.legend(facecolor="#1C2128", labelcolor=TEXT, fontsize=7.5)

    # ── Panel 3: Cleaned spectrum ────────────────────────────────────
    ax3.plot(freqs_d, mag_d, color=SPEC_D, linewidth=0.7, alpha=0.9)
    ax3.set_xlim(0, 1000)
    _style(ax3, "③ Cleaned Spectrum  — notches applied")
    ax3.set_xlabel("Frequency (Hz)", color=TEXT, fontsize=8)
    ax3.set_ylabel("Magnitude", color=TEXT, fontsize=8)

    # ── Panel 4: Restored waveform ───────────────────────────────────
    ax4.plot(denoised.t, denoised.samples, color=CLEAN_C, linewidth=0.6, alpha=0.85)
    ax4.plot(clean.t, clean.samples, color="white", linewidth=0.4,
             alpha=0.35, linestyle="--", label="ground truth")
    _style(ax4, "④ Restored Waveform  (time domain)")
    ax4.set_xlabel("Time (s)", color=TEXT, fontsize=8)
    ax4.set_ylabel("Amplitude", color=TEXT, fontsize=8)
    ax4.legend(facecolor="#1C2128", labelcolor=TEXT, fontsize=7.5)

    # ── Title ────────────────────────────────────────────────────────
    fig.suptitle(
        "SpectralClean — Audio Restoration Pipeline\n"
        f"Hum: {HUM_FREQ} Hz (+{HARMONICS} harmonics)   White noise σ={WHITE_STD}",
        color="white", fontsize=13, y=0.97
    )

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✓ 4-panel figure saved → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════
#  STEP 7b — Spectrum comparison plot
# ══════════════════════════════════════════════════════════════════════

def plot_spectrum_comparison(
    clean: Signal,
    noisy: Signal,
    denoised: Signal,
    save_path: str = "report/spectrum_comparison.png",
) -> plt.Figure:
    """Full-range spectrum overlay for noisy vs denoised."""
    _, fn, mn, _ = compute_spectrum(noisy)
    _, fd, md, _ = compute_spectrum(denoised)
    _, fc, mc, _ = compute_spectrum(clean)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor("#0D1117")

    data = [
        (fc, mc, "#69FF47", "Clean (reference)"),
        (fn, mn, "#FF6B6B", "Noisy (corrupted)"),
        (fd, md, "#00E5FF", "Denoised (restored)"),
    ]
    for ax, (freqs, mag, col, lbl) in zip(axes, data):
        ax.set_facecolor("#161B22")
        ax.semilogy(freqs, mag + 1e-12, color=col, linewidth=0.8)
        ax.set_title(lbl, color="white", fontsize=10)
        ax.set_xlabel("Frequency (Hz)", color="#C9D1D9", fontsize=8)
        ax.set_ylabel("Magnitude (log)", color="#C9D1D9", fontsize=8)
        ax.tick_params(colors="#C9D1D9")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363D")
        ax.grid(True, color="#21262D", alpha=0.6)
        ax.set_xlim(0, clean.fs / 2)

    fig.suptitle("Spectrum Comparison: Clean / Noisy / Denoised",
                 color="white", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    print(f"✓ Spectrum comparison saved → {save_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "═" * 56)
    print("   SpectralClean — Audio Restoration & Analysis Studio")
    print("═" * 56)

    # ── Step 1: Clean signal ──────────────────────────────────────────
    print("\n[1/7] Synthesising clean speech signal …")
    t0 = time.perf_counter()
    clean = make_clean_signal()
    print(f"      {clean}  ({time.perf_counter() - t0:.3f}s)")

    # ── Step 2: Noise injection ───────────────────────────────────────
    print("[2/7] Injecting hum + white noise …")
    noisy = inject_noise(clean)
    print(f"      Hum: {HUM_FREQ} Hz × {HARMONICS} harmonics, amp={HUM_AMP}")
    print(f"      White noise: σ={WHITE_STD}")

    # ── Step 3: Save audio files ──────────────────────────────────────
    print("[3/7] Writing .wav files …")
    to_wav(clean,  "report/clean.wav")
    to_wav(noisy,  "report/noisy.wav")
    print("      Saved: report/clean.wav, report/noisy.wav")

    # ── Step 4: Spectral analysis (visual check) ──────────────────────
    print("[4/7] Computing spectra via scratch FFT …")
    _, freqs, mag, N_padded = compute_spectrum(noisy)
    hum_bins = [int(round(HUM_FREQ * h * N_padded / FS)) for h in range(1, HARMONICS + 1)]
    print(f"      Padded N = {N_padded}  (next power of 2 ≥ {clean.N})")
    print(f"      Hum bins at harmonics: {hum_bins}")

    # ── Step 5: Denoising ─────────────────────────────────────────────
    print("[5/7] Denoising (band-stop masking + low-pass) …")
    t0 = time.perf_counter()
    denoised = denoise_combined(
        noisy,
        hum_freq_hz=HUM_FREQ,
        hum_bandwidth_hz=HUM_BW,
        harmonics=HARMONICS,
        lowpass_cutoff_hz=LP_CUTOFF,
    )
    print(f"      Denoising completed in {time.perf_counter() - t0:.3f}s")
    to_wav(denoised, "report/denoised.wav")
    print("      Saved: report/denoised.wav")

    # ── Step 6: Metrics ───────────────────────────────────────────────
    print("[6/7] Computing SNR metrics …")
    metrics = snr_improvement(clean, noisy, denoised)
    print_metrics_report(metrics, "SNR Improvement Report")

    # ── Step 7: Figures ───────────────────────────────────────────────
    print("[7/7] Generating figures …")

    fig_panel = plot_four_panel(clean, noisy, denoised)
    fig_spec   = plot_spectrum_comparison(clean, noisy, denoised)

    # Benchmark (Phase 2.4) — runs inline
    print("\n  Running DFT vs FFT benchmark …")
    from benchmark import run_benchmark, plot_benchmark
    bench_results = run_benchmark(n_min_exp=4, n_max_exp=14, dft_max_exp=11)
    fig_bench = plot_benchmark(bench_results, save_path="report/benchmark_dft_vs_fft.png")

    # ── Summary ───────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  Pipeline complete!  All outputs in report/")
    print("  ├── clean.wav")
    print("  ├── noisy.wav")
    print("  ├── denoised.wav")
    print("  ├── four_panel.png            ← main figure")
    print("  ├── spectrum_comparison.png")
    print("  └── benchmark_dft_vs_fft.png  ← benchmark figure")
    print(f"\n  SNR improvement: {metrics['improvement_db']:+.2f} dB")
    print("═" * 56 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
