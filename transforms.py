"""
transforms.py — Phase 2.1, 2.2, 2.3
Spectral transformation engine built from scratch.

Implements:
  - Naive O(N²) DFT  — direct summation
  - Radix-2 Cooley-Tukey FFT  — recursive O(N log N)
  - IFFT  — via conjugate trick
  - Zero-padding utility

No numpy.fft, scipy.fft, or any FFT library is used in the pipeline.
numpy.fft is only imported inside the `validate_against_numpy` helper,
which is explicitly for correctness checking only.
"""

import numpy as np
from signal_core import Signal


# ======================================================================
# Phase 2.1 — Naive DFT
# ======================================================================

def dft(x: np.ndarray) -> np.ndarray:
    """
    Compute the Discrete Fourier Transform of real/complex array x.

    Definition:
        X[k] = Σ_{n=0}^{N-1}  x[n] · e^{-j·2π·k·n / N}

    Complexity: O(N²)

    Implemented as a vectorised matrix multiplication for clarity
    and speed without sacrificing mathematical transparency.

    Parameters
    ----------
    x : 1-D array-like of length N

    Returns
    -------
    X : complex128 ndarray of length N
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    # Build the N×N DFT matrix W where W[k,n] = e^{-j2πkn/N}
    n = np.arange(N)              # shape (N,)
    k = n[:, np.newaxis]          # shape (N, 1) — broadcast rows
    # Twiddle factor matrix: W[k, n] = e^{-j2π k n / N}
    W = np.exp(-2j * np.pi * k * n / N)   # shape (N, N)
    return W @ x                           # matrix-vector product


def dft_slow(x: np.ndarray) -> np.ndarray:
    """
    Naive nested-loop DFT — pedagogically explicit.
    Equivalent to `dft` but shows the inner/outer sum directly.
    Only used to verify the vectorised version; not used in pipeline.
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X


# ======================================================================
# Phase 2.2 — Radix-2 Cooley-Tukey FFT (recursive)
# ======================================================================

def fft(x: np.ndarray) -> np.ndarray:
    """
    Cooley-Tukey Radix-2 DIT FFT.

    Divide-and-conquer strategy:
        X[k]       = E[k] + W_N^k · O[k]
        X[k + N/2] = E[k] − W_N^k · O[k]

    where E and O are the DFTs of the even- and odd-indexed sub-sequences.

    Recursion bottoms out at N=1 (trivial DFT).

    Constraint: N must be a power of 2.  Use `next_power_of_two` / 
    `zero_pad_to_power2` helpers below to satisfy this requirement.

    Complexity: O(N log N)

    Parameters
    ----------
    x : 1-D array of length N (must be a power of 2)

    Returns
    -------
    X : complex128 ndarray of length N
    """
    x = np.asarray(x, dtype=np.complex128)
    N = len(x)

    if N == 1:
        return x.copy()

    if N & (N - 1) != 0:
        raise ValueError(
            f"FFT requires N to be a power of 2, got N={N}. "
            "Use zero_pad_to_power2() before calling fft()."
        )

    # Split into even- and odd-indexed subsequences
    E = fft(x[::2])   # even indices: 0, 2, 4, …
    O = fft(x[1::2])  # odd  indices: 1, 3, 5, …

    # Twiddle factors W_N^k = e^{-j2πk/N} for k = 0 … N/2-1
    k = np.arange(N // 2)
    W = np.exp(-2j * np.pi * k / N)

    WO = W * O

    # Butterfly combination
    top    = E + WO          # X[k]         k = 0 … N/2-1
    bottom = E - WO          # X[k + N/2]   k = 0 … N/2-1

    return np.concatenate([top, bottom])


# ======================================================================
# Phase 2.3 — Inverse FFT (conjugate trick)
# ======================================================================

def ifft(X: np.ndarray) -> np.ndarray:
    """
    Inverse DFT using the conjugate trick.

    Derivation:
        x[n] = (1/N) · conj( FFT( conj(X) ) )

    This exploits the fact that the DFT and IDFT share the same
    butterfly structure — so we can reuse the forward FFT with
    conjugated inputs and a 1/N normalisation.

    Parameters
    ----------
    X : frequency-domain spectrum of length N (power of 2)

    Returns
    -------
    x : time-domain signal (complex128); for real inputs,
        imaginary part will be numerically negligible (< 1e-10).
    """
    X = np.asarray(X, dtype=np.complex128)
    N = len(X)
    # Conjugate → forward FFT → conjugate → normalise
    return np.conj(fft(np.conj(X))) / N


def ifft_real(X: np.ndarray) -> np.ndarray:
    """
    Convenience wrapper: returns real part of ifft (discards negligible
    imaginary component that arises from floating-point rounding).
    """
    return ifft(X).real


# ======================================================================
# Frequency-axis utilities
# ======================================================================

def fft_frequencies(N: int, fs: float) -> np.ndarray:
    """
    Compute the frequency axis for an N-point FFT at sampling rate fs.

    Returns the positive-only axis (0 … fs/2) — the physically
    meaningful half for real signals.

    Returns
    -------
    freqs : ndarray of length N//2 + 1
    """
    return np.arange(N // 2 + 1) * fs / N


def fft_magnitude(X: np.ndarray) -> np.ndarray:
    """
    One-sided magnitude spectrum normalised by N.
    Doubles the one-sided bins (except DC and Nyquist) so that
    the displayed amplitude matches the original signal amplitude.
    """
    N = len(X)
    mag = np.abs(X) / N
    one_sided = mag[: N // 2 + 1]
    one_sided[1:-1] *= 2  # account for negative frequencies
    return one_sided


# ======================================================================
# Zero-padding helpers
# ======================================================================

def next_power_of_two(n: int) -> int:
    """Return the smallest power of 2 that is ≥ n."""
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p <<= 1
    return p


def zero_pad_to_power2(x: np.ndarray) -> np.ndarray:
    """
    Zero-pad array x to the next power of 2.
    Returns x unchanged if len(x) is already a power of 2.
    """
    N = next_power_of_two(len(x))
    if N == len(x):
        return x.astype(np.complex128)
    padded = np.zeros(N, dtype=np.complex128)
    padded[: len(x)] = x
    return padded


# ======================================================================
# Signal-level wrappers
# ======================================================================

def signal_fft(sig: Signal):
    """
    Compute the FFT of a Signal.
    Returns (X, freqs, mag) — spectrum, frequency axis, magnitude.
    """
    x_padded = zero_pad_to_power2(sig.samples)
    N = len(x_padded)
    X = fft(x_padded)
    freqs = fft_frequencies(N, sig.fs)
    mag = fft_magnitude(X)
    return X, freqs, mag


def signal_ifft(X: np.ndarray, fs: float, original_N: int) -> Signal:
    """
    Reconstruct a time-domain Signal from its spectrum X.
    Trims the output to `original_N` samples (removes zero-padding).
    """
    x_rec = ifft_real(X)
    return Signal(x_rec[:original_N], fs)


# ======================================================================
# Correctness validation (TESTING ONLY — never called in pipeline)
# ======================================================================

def validate_against_numpy(x: np.ndarray, tol: float = 1e-8) -> dict:
    """
    Compare scratch DFT & FFT output against numpy.fft.fft.
    Called ONLY during unit tests; never imported in main pipeline.

    Returns a dict with max absolute errors and pass/fail booleans.
    """
    import numpy.fft as npfft  # allowed ONLY for validation

    x = np.asarray(x, dtype=np.complex128)
    x_padded = zero_pad_to_power2(x)

    ref = npfft.fft(x_padded)
    scratch_fft_out = fft(x_padded)
    scratch_dft_out = dft(x_padded[:min(512, len(x_padded))])  # DFT too slow for large N

    fft_error = np.max(np.abs(scratch_fft_out - ref))
    dft_ref   = npfft.fft(x_padded[:min(512, len(x_padded))])
    dft_error = np.max(np.abs(scratch_dft_out - dft_ref[:len(scratch_dft_out)]))

    return {
        "fft_max_error": fft_error,
        "fft_pass":      fft_error < tol,
        "dft_max_error": dft_error,
        "dft_pass":      dft_error < tol,
        "N_padded":      len(x_padded),
    }
