"""
signal_core.py — Phase 1.1 & 1.2
Signal representation and elementary LTI operations.

Constraints: NumPy + Matplotlib only (no scipy.signal).
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from dataclasses import dataclass, field


# ======================================================================
# Effect Chain Configuration (Phase 2 — v2)
# ======================================================================

@dataclass
class EffectChainConfig:
    """
    Parameter container for the unified audio effect chain.

    Processing order (when enabled):
        Noise Removal → Equalizer → Echo → Reverb

    Validation runs in __post_init__ — raises ValueError on unsafe params.
    """
    # Spectral subtraction noise removal
    noise_removal: bool = False
    noise_start_s: float = 0.0
    noise_end_s: float = 0.2
    noise_over_subtraction: float = 1.0

    # 5-band graphic EQ: gains in dB for [60, 230, 910, 4000, 14000] Hz
    eq_enabled: bool = False
    eq_gains_db: list = field(default_factory=lambda: [0.0, 0.0, 0.0, 0.0, 0.0])

    # Echo / Delay
    echo_enabled: bool = False
    echo_delay_ms: float = 200.0
    echo_feedback: float = 0.30    # MUST be < 1.0
    echo_wet: float = 0.50

    # Algorithmic reverb
    reverb_enabled: bool = False
    reverb_room_size: float = 0.50  # 0–1
    reverb_damping: float = 0.50    # 0–1
    reverb_wet: float = 0.33

    # Convolution reverb (overrides algorithmic when set)
    rir_signal: object = None       # Signal or None

    def __post_init__(self):
        if self.echo_feedback >= 1.0:
            raise ValueError(
                f"echo_feedback must be < 1.0 to prevent instability, "
                f"got {self.echo_feedback}"
            )
        if not (0.0 <= self.echo_wet <= 1.0):
            raise ValueError(f"echo_wet must be in [0, 1], got {self.echo_wet}")
        if not (0.0 <= self.reverb_room_size <= 1.0):
            raise ValueError(
                f"reverb_room_size must be in [0, 1], got {self.reverb_room_size}"
            )
        if not (0.0 <= self.reverb_wet <= 1.0):
            raise ValueError(f"reverb_wet must be in [0, 1], got {self.reverb_wet}")
        if len(self.eq_gains_db) != 5:
            raise ValueError(
                f"eq_gains_db must have exactly 5 values (one per EQ band), "
                f"got {len(self.eq_gains_db)}"
            )


class Signal:
    """
    Discrete-time signal container.

    Attributes
    ----------
    samples : np.ndarray  — 1-D float64 sample array
    fs      : float       — sampling rate in Hz
    t       : np.ndarray  — time vector t[n] = n / fs
    """

    def __init__(self, samples: np.ndarray, fs: float):
        """
        Parameters
        ----------
        samples : array-like  — raw sample values
        fs      : float       — sampling rate (Hz), must be > 0
        """
        if fs <= 0:
            raise ValueError(f"Sampling rate must be positive, got {fs}")
        self.samples = np.array(samples, dtype=np.float64)
        self.fs = float(fs)
        self.t = np.arange(len(self.samples)) / self.fs

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def N(self) -> int:
        """Number of samples."""
        return len(self.samples)

    @property
    def duration(self) -> float:
        """Signal duration in seconds."""
        return self.N / self.fs

    # ------------------------------------------------------------------
    # 1.2 Elementary operations
    # ------------------------------------------------------------------
    def shift(self, k: int) -> "Signal":
        """
        Integer shift by k samples.
        Positive k → shift right (delay); negative k → shift left (advance).
        Edges are zero-padded to preserve length.
        """
        result = np.zeros(self.N, dtype=np.float64)
        if k >= 0:
            # delay: samples[0] lands at index k
            end = min(self.N, self.N - k)
            if end > 0:
                result[k:k + end] = self.samples[:end]
        else:
            # advance: samples[-k] lands at index 0
            start = -k
            result[: self.N - start] = self.samples[start:]
        return Signal(result, self.fs)

    def scale(self, alpha: float) -> "Signal":
        """Amplitude scale: y[n] = alpha * x[n]."""
        return Signal(self.samples * alpha, self.fs)

    def time_scale(self, factor: float) -> "Signal":
        """
        Time-domain resampling by integer-nearest factor.
        factor > 1 → expansion (slower); factor < 1 → compression (faster).
        Uses nearest-neighbour interpolation so no scipy needed.
        """
        new_N = max(1, int(round(self.N * factor)))
        indices = np.clip(
            (np.arange(new_N) / factor).astype(int), 0, self.N - 1
        )
        return Signal(self.samples[indices], self.fs)

    def add(self, other: "Signal") -> "Signal":
        """
        Add two signals sample-by-sample.
        Length mismatch is handled by zero-padding the shorter signal.
        Both signals must share the same sampling rate.
        """
        self._check_fs(other)
        n = max(self.N, other.N)
        a = _zero_pad_to(self.samples, n)
        b = _zero_pad_to(other.samples, n)
        return Signal(a + b, self.fs)

    def multiply(self, other: "Signal") -> "Signal":
        """
        Element-wise multiply two signals.
        Zero-pads shorter signal.
        """
        self._check_fs(other)
        n = max(self.N, other.N)
        a = _zero_pad_to(self.samples, n)
        b = _zero_pad_to(other.samples, n)
        return Signal(a * b, self.fs)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot(
        self,
        ax: plt.Axes = None,
        title: str = "Signal (time domain)",
        color: str = "#4FC3F7",
        alpha: float = 0.85,
        label: str = None,
    ) -> plt.Axes:
        """
        Plot the signal in the time domain.

        Parameters
        ----------
        ax    : existing Axes to draw on (creates new figure if None)
        title : axes title string
        color : line colour (hex or named)
        alpha : line opacity

        Returns
        -------
        ax : the Axes object drawn on
        """
        created = ax is None
        if created:
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#1A1A2E")

        ax.set_facecolor("#16213E")
        ax.plot(self.t, self.samples, color=color, alpha=alpha,
                linewidth=0.8, label=label)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.set_xlabel("Time (s)", color="#B0BEC5")
        ax.set_ylabel("Amplitude", color="#B0BEC5")
        ax.tick_params(colors="#B0BEC5")
        for spine in ax.spines.values():
            spine.set_edgecolor("#37474F")
        if label:
            ax.legend(facecolor="#1A1A2E", labelcolor="white", fontsize=8)

        if created:
            plt.tight_layout()

        return ax

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"Signal(N={self.N}, fs={self.fs:.1f} Hz, "
            f"duration={self.duration:.4f} s)"
        )

    def __len__(self) -> int:
        return self.N

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _check_fs(self, other: "Signal") -> None:
        if self.fs != other.fs:
            raise ValueError(
                f"Sampling rates must match: {self.fs} vs {other.fs}"
            )


# ------------------------------------------------------------------
# Module-level utilities
# ------------------------------------------------------------------
def _zero_pad_to(arr: np.ndarray, n: int) -> np.ndarray:
    """Return arr zero-padded (or truncated) to length n."""
    if len(arr) >= n:
        return arr[:n]
    return np.concatenate([arr, np.zeros(n - len(arr))])


def from_wav(path: str) -> Signal:
    """
    Load a .wav file into a Signal.
    Uses scipy.io.wavfile **only** for I/O — no math.

    Multi-channel audio is averaged to mono.
    """
    from scipy.io import wavfile  # I/O only — explicitly allowed

    fs, data = wavfile.read(path)
    data = data.astype(np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)  # stereo → mono
    # Normalise integer PCM to [-1, 1]
    if data.dtype != np.float64:
        data = data.astype(np.float64)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return Signal(data, float(fs))


def to_wav(sig: Signal, path: str, bit_depth: int = 16) -> None:
    """
    Write Signal to a .wav file.
    scipy.io.wavfile used for I/O only — no math.
    """
    from scipy.io import wavfile  # I/O only — explicitly allowed

    dtype_map = {16: np.int16, 32: np.int32}
    if bit_depth not in dtype_map:
        raise ValueError("bit_depth must be 16 or 32")
    scale = np.iinfo(dtype_map[bit_depth]).max
    out = np.clip(sig.samples, -1.0, 1.0)
    out = (out * scale).astype(dtype_map[bit_depth])
    wavfile.write(path, int(sig.fs), out)


def sine_wave(freq: float, duration: float, fs: float, amplitude: float = 1.0, phase: float = 0.0) -> Signal:
    """Convenience: generate a pure sine wave Signal."""
    t = np.arange(int(duration * fs)) / fs
    samples = amplitude * np.sin(2 * np.pi * freq * t + phase)
    return Signal(samples, fs)


def white_noise(duration: float, fs: float, std: float = 0.05, seed: int = 42) -> Signal:
    """Convenience: generate Gaussian white noise Signal."""
    rng = np.random.default_rng(seed)
    n = int(duration * fs)
    samples = rng.normal(0.0, std, n)
    return Signal(samples, fs)


def unit_impulse(N: int, fs: float, k: int = 0) -> Signal:
    """Unit impulse δ[n-k] of length N."""
    samples = np.zeros(N)
    if 0 <= k < N:
        samples[k] = 1.0
    return Signal(samples, fs)
