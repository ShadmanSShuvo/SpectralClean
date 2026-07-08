"""
convolution.py — Phase 1.3
Direct (linear) convolution implemented from the summation formula.

    y[n] = Σ_{k=0}^{M-1}  h[k] · x[n - k]

No numpy.convolve or scipy is used for the computation itself.
"""

import numpy as np
from signal_core import Signal


def convolve(x: Signal, h: Signal) -> Signal:
    """
    Linear convolution of signal x with kernel h.

    Implements the direct sum:
        y[n] = Σ_{k} h[k] · x[n - k]

    Output length: N_x + N_h - 1  (full linear convolution).

    Parameters
    ----------
    x : input Signal
    h : impulse-response / kernel Signal (must share fs with x)

    Returns
    -------
    Signal — convolved result at the same sampling rate
    """
    if x.fs != h.fs:
        raise ValueError(
            f"Signals must share sampling rate: {x.fs} vs {h.fs}"
        )

    x_arr = x.samples
    h_arr = h.samples
    Nx = len(x_arr)
    Nh = len(h_arr)
    Ny = Nx + Nh - 1

    y = np.zeros(Ny, dtype=np.float64)

    # Direct summation — the textbook definition
    for n in range(Ny):
        total = 0.0
        # k ranges where both h[k] and x[n-k] are in-bounds
        k_lo = max(0, n - (Nx - 1))
        k_hi = min(Nh - 1, n)
        for k in range(k_lo, k_hi + 1):
            total += h_arr[k] * x_arr[n - k]
        y[n] = total

    return Signal(y, x.fs)


def convolve_fast(x: Signal, h: Signal) -> Signal:
    """
    Vectorised convolution using matrix-vector multiplication.
    Mathematically identical to `convolve`; avoids Python-level loop.
    Used to speed up longer signals while keeping the direct-sum spirit.

    Constructs the Toeplitz-like matrix of h and multiplies by x.
    """
    if x.fs != h.fs:
        raise ValueError(
            f"Signals must share sampling rate: {x.fs} vs {h.fs}"
        )

    x_arr = x.samples
    h_arr = h.samples
    Nx = len(x_arr)
    Nh = len(h_arr)
    Ny = Nx + Nh - 1

    # Build input matrix: each column is a shifted version of h
    # Shape: (Ny, Nx) — effectively the convolution matrix
    conv_matrix = np.zeros((Ny, Nx), dtype=np.float64)
    for k, hk in enumerate(h_arr):
        conv_matrix[k: k + Nx, np.arange(Nx)] += hk * np.eye(Nx, dtype=np.float64)[0]

    # Simpler vectorised approach: sliding dot products
    y = np.zeros(Ny, dtype=np.float64)
    x_padded = np.concatenate([np.zeros(Nh - 1), x_arr, np.zeros(Nh - 1)])
    h_flipped = h_arr[::-1]  # flip h for correlation-style sliding

    for n in range(Ny):
        y[n] = np.dot(x_padded[n: n + Nh], h_flipped)

    return Signal(y, x.fs)


def moving_average_kernel(M: int, fs: float) -> Signal:
    """
    Create a normalised moving-average (box) kernel of length M.

    h[n] = 1/M  for 0 ≤ n < M
    """
    return Signal(np.ones(M, dtype=np.float64) / M, fs)


def fir_lowpass_kernel(cutoff_hz: float, fs: float, num_taps: int = 51) -> Signal:
    """
    Design a simple windowed-sinc FIR low-pass kernel.
    Uses a Hann window; no scipy involved.

    Parameters
    ----------
    cutoff_hz : cutoff frequency in Hz
    fs        : sampling rate
    num_taps  : filter length (odd number recommended)
    """
    if num_taps % 2 == 0:
        num_taps += 1  # enforce odd

    M = num_taps - 1
    half = M // 2
    fc = cutoff_hz / fs  # normalised cutoff [0, 0.5]

    n = np.arange(num_taps, dtype=np.float64)

    # Sinc kernel
    h = np.where(
        n == half,
        2 * fc,
        np.sin(2 * np.pi * fc * (n - half)) / (np.pi * (n - half))
    )

    # Hann window
    w = 0.5 * (1 - np.cos(2 * np.pi * n / M))
    h *= w
    h /= h.sum()  # normalise to unit DC gain

    return Signal(h, fs)
