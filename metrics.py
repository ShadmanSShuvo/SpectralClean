"""
metrics.py — Phase 4.1
SNR calculation and quantitative denoising metrics.

All math is pure NumPy. No scipy.
"""

import numpy as np
from signal_core import Signal


def signal_power(sig: Signal) -> float:
    """
    Compute mean power of a signal.
        P = (1/N) · Σ x[n]²
    """
    return float(np.mean(sig.samples ** 2))


def snr_db(clean: Signal, noisy: Signal) -> float:
    """
    Signal-to-Noise Ratio in dB.

    Given a clean reference and a noisy (or processed) version:
        noise = noisy - clean  (residual)
        SNR   = 10 · log10( P_signal / P_noise )

    A higher SNR means less noise relative to signal.

    Parameters
    ----------
    clean : ground-truth clean signal
    noisy : noisy or processed signal (same length as clean)

    Returns
    -------
    snr : float — SNR in dB (may be -inf if signal power is zero)
    """
    if clean.N != noisy.N:
        # Trim to shortest
        n = min(clean.N, noisy.N)
        c = Signal(clean.samples[:n], clean.fs)
        d = Signal(noisy.samples[:n], noisy.fs)
    else:
        c, d = clean, noisy

    p_signal = signal_power(c)
    residual = d.samples - c.samples
    p_noise  = float(np.mean(residual ** 2))

    if p_noise == 0:
        return float("inf")
    if p_signal == 0:
        return float("-inf")

    return 10.0 * np.log10(p_signal / p_noise)


def psnr_db(clean: Signal, processed: Signal) -> float:
    """
    Peak Signal-to-Noise Ratio in dB.

    PSNR = 20 · log10( peak / RMSE )
    where peak = max absolute value of the clean signal.
    """
    n = min(clean.N, processed.N)
    c = clean.samples[:n]
    d = processed.samples[:n]

    peak = np.max(np.abs(c))
    if peak == 0:
        return float("-inf")

    mse = np.mean((c - d) ** 2)
    if mse == 0:
        return float("inf")

    return 20.0 * np.log10(peak / np.sqrt(mse))


def rmse(clean: Signal, processed: Signal) -> float:
    """Root Mean Square Error between clean and processed signals."""
    n = min(clean.N, processed.N)
    return float(np.sqrt(np.mean((clean.samples[:n] - processed.samples[:n]) ** 2)))


def noise_power_estimate(sig: Signal, clean: Signal) -> float:
    """Estimate noise power as the mean squared residual (sig - clean)."""
    n = min(sig.N, clean.N)
    residual = sig.samples[:n] - clean.samples[:n]
    return float(np.mean(residual ** 2))


def snr_improvement(clean: Signal, noisy: Signal, denoised: Signal) -> dict:
    """
    Compute a full SNR improvement report.

    Returns
    -------
    dict with keys:
        snr_before  — SNR of noisy vs clean  (dB)
        snr_after   — SNR of denoised vs clean (dB)
        improvement — difference in dB (positive = improvement)
        psnr_before, psnr_after
        rmse_before, rmse_after
    """
    return {
        "snr_before_db":  snr_db(clean, noisy),
        "snr_after_db":   snr_db(clean, denoised),
        "improvement_db": snr_db(clean, denoised) - snr_db(clean, noisy),
        "psnr_before_db": psnr_db(clean, noisy),
        "psnr_after_db":  psnr_db(clean, denoised),
        "rmse_before":    rmse(clean, noisy),
        "rmse_after":     rmse(clean, denoised),
    }


def print_metrics_report(metrics: dict, label: str = "SNR Report") -> None:
    """Pretty-print the metrics dictionary."""
    w = 44
    print(f"\n{'═' * w}")
    print(f"  {label}")
    print(f"{'═' * w}")
    print(f"  {'SNR before:':<28} {metrics['snr_before_db']:+.2f} dB")
    print(f"  {'SNR after:':<28} {metrics['snr_after_db']:+.2f} dB")
    print(f"  {'SNR improvement:':<28} {metrics['improvement_db']:+.2f} dB")
    print(f"  {'PSNR before:':<28} {metrics['psnr_before_db']:+.2f} dB")
    print(f"  {'PSNR after:':<28} {metrics['psnr_after_db']:+.2f} dB")
    print(f"  {'RMSE before:':<28} {metrics['rmse_before']:.6f}")
    print(f"  {'RMSE after:':<28} {metrics['rmse_after']:.6f}")
    print(f"{'═' * w}\n")
