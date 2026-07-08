"""
filters.py — Phase 3.3
Frequency-domain filters operating entirely on scratch FFT/IFFT.

Provides:
  - band_stop_mask   — zero-out bins near a target frequency
  - low_pass_mask    — keep bins below cutoff
  - high_pass_mask   — keep bins above cutoff
  - apply_spectral_filter — full pipeline: FFT → mask → IFFT
  - convolution-based FIR low-pass (alternate method)
"""

import numpy as np
from signal_core import Signal
from transforms import fft, ifft_real, zero_pad_to_power2, fft_frequencies, signal_ifft


# ======================================================================
# Frequency-bin masking helpers
# ======================================================================

def _bin_index(freq_hz: float, N: int, fs: float) -> int:
    """Convert a physical frequency (Hz) to its FFT bin index."""
    return int(round(freq_hz * N / fs))


def band_stop_mask(
    N: int,
    fs: float,
    center_hz: float,
    bandwidth_hz: float = 5.0,
) -> np.ndarray:
    """
    Create a real-valued mask array (length N) that zeros out bins
    within ±(bandwidth_hz/2) of center_hz.

    The mask is applied symmetrically to both positive and negative
    frequency halves to preserve a real-valued time-domain output.

    Returns
    -------
    mask : ndarray of shape (N,) with values in {0, 1}
    """
    mask = np.ones(N, dtype=np.float64)
    half_bw_bins = max(1, int(round(bandwidth_hz / 2 * N / fs)))
    center_bin   = _bin_index(center_hz, N, fs)

    lo = max(0,     center_bin - half_bw_bins)
    hi = min(N // 2, center_bin + half_bw_bins)

    mask[lo: hi + 1] = 0.0

    # Mirror to negative frequencies (conjugate symmetry)
    if lo > 0:
        mask[N - hi: N - lo + 1] = 0.0

    return mask


def low_pass_mask(N: int, fs: float, cutoff_hz: float) -> np.ndarray:
    """
    Low-pass mask: keep bins with |f| ≤ cutoff_hz.
    """
    mask = np.zeros(N, dtype=np.float64)
    cutoff_bin = _bin_index(cutoff_hz, N, fs)
    mask[: cutoff_bin + 1] = 1.0
    # Mirror for negative frequencies
    mask[N - cutoff_bin:] = 1.0
    return mask


def high_pass_mask(N: int, fs: float, cutoff_hz: float) -> np.ndarray:
    """
    High-pass mask: keep bins with |f| ≥ cutoff_hz.
    (Complement of low-pass mask.)
    """
    return 1.0 - low_pass_mask(N, fs, cutoff_hz)


def band_pass_mask(
    N: int,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> np.ndarray:
    """
    Band-pass mask: keep bins with low_hz ≤ |f| ≤ high_hz.
    """
    lp = low_pass_mask(N, fs, high_hz)
    hp = high_pass_mask(N, fs, low_hz)
    return lp * hp


# ======================================================================
# Main spectral filtering pipeline
# ======================================================================

def apply_spectral_filter(
    sig: Signal,
    mask: np.ndarray,
) -> Signal:
    """
    Apply a frequency-domain mask to a signal.

    Pipeline:
        1. Zero-pad x to next power of 2
        2. Compute scratch FFT(x) → X[k]
        3. Apply element-wise mask: X_filtered[k] = mask[k] · X[k]
        4. Compute scratch IFFT(X_filtered) → x_reconstructed
        5. Trim back to original length

    Parameters
    ----------
    sig  : input Signal
    mask : ndarray of length N_padded (values in [0, 1])

    Returns
    -------
    Signal — filtered, same length and fs as input
    """
    from transforms import fft, ifft_real, zero_pad_to_power2

    original_N = sig.N
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)

    if len(mask) != N:
        raise ValueError(
            f"Mask length {len(mask)} must match padded FFT length {N}."
        )

    X = fft(x_padded)
    X_filtered = X * mask
    x_rec = ifft_real(X_filtered)

    return Signal(x_rec[:original_N], sig.fs)


def denoise_hum(
    sig: Signal,
    hum_freq_hz: float,
    bandwidth_hz: float = 5.0,
    harmonics: int = 3,
) -> Signal:
    """
    Remove a periodic hum and its harmonics using band-stop masking.

    Parameters
    ----------
    sig          : noisy input Signal
    hum_freq_hz  : fundamental hum frequency in Hz (e.g., 50 or 60 Hz)
    bandwidth_hz : notch width per harmonic in Hz
    harmonics    : number of harmonics to suppress (1 = fundamental only)

    Returns
    -------
    Signal — hum-suppressed signal
    """
    original_N = sig.N
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    fs = sig.fs

    mask = np.ones(N, dtype=np.float64)
    for h in range(1, harmonics + 1):
        freq = hum_freq_hz * h
        if freq < fs / 2:
            notch = band_stop_mask(N, fs, freq, bandwidth_hz)
            mask *= notch  # combine notches multiplicatively

    return apply_spectral_filter(sig, mask)


def denoise_lowpass(sig: Signal, cutoff_hz: float) -> Signal:
    """
    Low-pass filter: remove all content above cutoff_hz.
    Typically used to reduce broadband (white) noise floor.
    """
    original_N = sig.N
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    mask = low_pass_mask(N, sig.fs, cutoff_hz)
    return apply_spectral_filter(sig, mask)


def denoise_combined(
    sig: Signal,
    hum_freq_hz: float,
    hum_bandwidth_hz: float = 5.0,
    harmonics: int = 3,
    lowpass_cutoff_hz: float = None,
) -> Signal:
    """
    Combined denoising: band-stop hum notch + optional low-pass.

    This is the main denoising entry-point used in main.py.
    """
    original_N = sig.N
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    fs = sig.fs

    # Start with all-pass mask
    mask = np.ones(N, dtype=np.float64)

    # Band-stop notches for hum harmonics
    for h in range(1, harmonics + 1):
        freq = hum_freq_hz * h
        if freq < fs / 2:
            notch = band_stop_mask(N, fs, freq, hum_bandwidth_hz)
            mask *= notch

    # Optional high-frequency attenuation
    if lowpass_cutoff_hz is not None:
        lp = low_pass_mask(N, fs, lowpass_cutoff_hz)
        mask *= lp

    return apply_spectral_filter(sig, mask)


# ======================================================================
# FIR convolution-based low-pass (alternate method — Phase 3.3)
# ======================================================================

def fir_lowpass_convolution(sig: Signal, cutoff_hz: float, num_taps: int = 101) -> Signal:
    """
    Alternate low-pass method using the convolution-based FIR approach
    from convolution.py.

    This demonstrates the time-domain filtering path vs the masking path,
    both yielding equivalent results for ideal brick-wall masks.
    """
    from convolution import fir_lowpass_kernel, convolve_fast

    kernel = fir_lowpass_kernel(cutoff_hz, sig.fs, num_taps)
    filtered = convolve_fast(sig, kernel)

    # Convolve adds Nh-1 samples; trim to original length
    trimmed = Signal(filtered.samples[num_taps // 2: num_taps // 2 + sig.N], sig.fs)
    return trimmed
