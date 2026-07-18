"""
effects.py — Phase 5 (SpectralClean v2)
Audio effects: Equalizer, Echo/Delay, Reverb, Spectral Subtraction.

ALL DSP implemented from scratch using NumPy only.
No scipy.signal, scipy.fft, or numpy.fft used anywhere in this module.
"""

import numpy as np
from signal_core import Signal
from transforms import fft, ifft_real, zero_pad_to_power2


# ======================================================================
# Biquad IIR filter primitives
# ======================================================================

def _biquad_filter(x: np.ndarray, b0: float, b1: float, b2: float,
                   a1: float, a2: float) -> np.ndarray:
    """
    Apply a single second-order IIR (biquad) section to array x.

    Difference equation:
        y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2]
                       − a1·y[n-1] − a2·y[n-2]

    Parameters are the Direct Form I coefficients (a0 = 1 assumed).
    No scipy.signal.lfilter used — pure sample-by-sample loop.
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float64)
    x1 = x2 = 0.0   # x[n-1], x[n-2]
    y1 = y2 = 0.0   # y[n-1], y[n-2]
    for n in range(N):
        xn = x[n]
        yn = b0 * xn + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, xn
        y2, y1 = y1, yn
        y[n] = yn
    return y


# ------------------------------------------------------------------
# Biquad coefficient calculators (Audio EQ Cookbook derivations)
# ------------------------------------------------------------------

def _biquad_lowpass(cutoff_hz: float, fs: float, Q: float = 0.707):
    """2nd-order Butterworth low-pass biquad coefficients."""
    w0 = 2.0 * np.pi * cutoff_hz / fs
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * Q)
    b1_val = 1.0 - cos_w0
    b0 = b1_val / 2.0
    b1 = b1_val
    b2 = b1_val / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_highpass(cutoff_hz: float, fs: float, Q: float = 0.707):
    """2nd-order Butterworth high-pass biquad coefficients."""
    w0 = 2.0 * np.pi * cutoff_hz / fs
    cos_w0 = np.cos(w0)
    alpha = np.sin(w0) / (2.0 * Q)
    b0 = (1.0 + cos_w0) / 2.0
    b1 = -(1.0 + cos_w0)
    b2 = (1.0 + cos_w0) / 2.0
    a0 = 1.0 + alpha
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_peaking(center_hz: float, fs: float, gain_db: float,
                    Q: float = 1.0):
    """
    Peaking EQ biquad — boosts/cuts gain_db dB around center_hz.
    gain_db > 0 → boost; gain_db < 0 → cut.
    """
    if abs(gain_db) < 1e-6:
        return 1.0, 0.0, 0.0, 0.0, 0.0   # pass-through
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * center_hz / fs
    alpha = np.sin(w0) / (2.0 * Q)
    cos_w0 = np.cos(w0)
    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_lowshelf(shelf_hz: float, fs: float, gain_db: float,
                     S: float = 1.0):
    """Low-shelf biquad (boosts/cuts all content below shelf_hz)."""
    if abs(gain_db) < 1e-6:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * shelf_hz / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
    sq = 2.0 * np.sqrt(A) * alpha
    b0 =      A * ((A + 1) - (A - 1) * cos_w0 + sq)
    b1 =  2 * A * ((A - 1) - (A + 1) * cos_w0)
    b2 =      A * ((A + 1) - (A - 1) * cos_w0 - sq)
    a0 =          (A + 1) + (A - 1) * cos_w0 + sq
    a1 =     -2 * ((A - 1) + (A + 1) * cos_w0)
    a2 =          (A + 1) + (A - 1) * cos_w0 - sq
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


def _biquad_highshelf(shelf_hz: float, fs: float, gain_db: float,
                      S: float = 1.0):
    """High-shelf biquad (boosts/cuts all content above shelf_hz)."""
    if abs(gain_db) < 1e-6:
        return 1.0, 0.0, 0.0, 0.0, 0.0
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * shelf_hz / fs
    cos_w0 = np.cos(w0)
    sin_w0 = np.sin(w0)
    alpha = sin_w0 / 2.0 * np.sqrt((A + 1.0 / A) * (1.0 / S - 1.0) + 2.0)
    sq = 2.0 * np.sqrt(A) * alpha
    b0 =      A * ((A + 1) + (A - 1) * cos_w0 + sq)
    b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
    b2 =      A * ((A + 1) + (A - 1) * cos_w0 - sq)
    a0 =          (A + 1) - (A - 1) * cos_w0 + sq
    a1 =      2 * ((A - 1) - (A + 1) * cos_w0)
    a2 =          (A + 1) - (A - 1) * cos_w0 - sq
    return b0 / a0, b1 / a0, b2 / a0, a1 / a0, a2 / a0


# ======================================================================
# Equalizer
# ======================================================================

# Standard 5-band graphic EQ centre frequencies (Hz)
EQ_BANDS_HZ = [60.0, 230.0, 910.0, 4000.0, 14000.0]


class Equalizer:
    """
    Parametric / graphic equalizer built from cascaded biquad stages.

    Usage — graphic mode (5 bands):
        eq = Equalizer.graphic_eq(gains_db=[0, +3, -6, +3, 0], fs=8000)
        out = eq.apply(sig)

    Usage — parametric mode:
        eq = Equalizer([
            ('peaking',   200, +6, 1.0),
            ('lowpass',  3500,  0, 0.7),
        ], fs=44100)
        out = eq.apply(sig)

    Band spec tuples: (filter_type, freq_hz, gain_db, Q)
        filter_type: 'lowpass' | 'highpass' | 'peaking' | 'lowshelf' | 'highshelf'
        gain_db is only used for peaking/shelf types.
    """

    def __init__(self, bands: list, fs: float):
        """
        Parameters
        ----------
        bands : list of (filter_type, freq_hz, gain_db, Q) tuples
        fs    : sampling rate in Hz
        """
        self.fs = fs
        self._stages = []   # each entry is (b0, b1, b2, a1, a2)
        for spec in bands:
            ftype, freq, gain_db, Q = spec
            coeffs = self._make_coeffs(ftype, freq, gain_db, Q)
            self._stages.append(coeffs)

    def _make_coeffs(self, ftype: str, freq: float, gain_db: float, Q: float):
        fs = self.fs
        if ftype == 'lowpass':
            return _biquad_lowpass(freq, fs, Q)
        elif ftype == 'highpass':
            return _biquad_highpass(freq, fs, Q)
        elif ftype == 'peaking':
            return _biquad_peaking(freq, fs, gain_db, Q)
        elif ftype == 'lowshelf':
            return _biquad_lowshelf(freq, fs, gain_db)
        elif ftype == 'highshelf':
            return _biquad_highshelf(freq, fs, gain_db)
        else:
            raise ValueError(f"Unknown filter type: {ftype!r}")

    def apply(self, sig: Signal) -> Signal:
        """Apply all EQ stages in cascade to the input signal."""
        x = sig.samples.copy()
        for (b0, b1, b2, a1, a2) in self._stages:
            x = _biquad_filter(x, b0, b1, b2, a1, a2)
        return Signal(x, sig.fs)

    @classmethod
    def graphic_eq(cls, gains_db: list, fs: float,
                   bands_hz: list = None) -> "Equalizer":
        """
        Build a 5-band (or N-band) graphic EQ from a list of dB gains.

        Parameters
        ----------
        gains_db  : list of gain values in dB (positive = boost, negative = cut)
        fs        : sampling rate
        bands_hz  : list of centre frequencies (defaults to EQ_BANDS_HZ)

        Returns
        -------
        Equalizer instance ready to call .apply(sig)
        """
        if bands_hz is None:
            bands_hz = EQ_BANDS_HZ
        if len(gains_db) != len(bands_hz):
            raise ValueError(
                f"gains_db length {len(gains_db)} must match "
                f"bands_hz length {len(bands_hz)}"
            )
        bands = [('peaking', hz, g, 1.0)
                 for hz, g in zip(bands_hz, gains_db)]
        return cls(bands, fs)


# ======================================================================
# Echo / Delay
# ======================================================================

def apply_echo(
    sig: Signal,
    delay_ms: float = 200.0,
    feedback: float = 0.35,
    wet: float = 0.5,
) -> Signal:
    """
    Single-tap feedback echo (comb filter):

        y[n] = x[n] + feedback · y[n - M]

    where M = delay_ms / 1000 * fs (integer sample count).

    Parameters
    ----------
    sig       : input Signal
    delay_ms  : delay time in milliseconds
    feedback  : echo feedback gain (must be < 1.0 to avoid instability)
    wet       : dry/wet mix (0 = dry only, 1 = wet only)

    Returns
    -------
    Signal — echo-processed output
    """
    if feedback >= 1.0:
        raise ValueError(
            f"Echo feedback must be < 1.0 to prevent instability, got {feedback}"
        )
    if not (0.0 <= wet <= 1.0):
        raise ValueError(f"wet must be in [0, 1], got {wet}")

    fs = sig.fs
    M = max(1, int(round(delay_ms / 1000.0 * fs)))
    x = sig.samples
    N = len(x)

    y = np.zeros(N, dtype=np.float64)
    for n in range(N):
        dry_sample = x[n]
        echo_sample = y[n - M] if n >= M else 0.0
        y[n] = dry_sample + feedback * echo_sample

    # Dry/wet blend
    mixed = (1.0 - wet) * x + wet * y
    return Signal(mixed, fs)


# ======================================================================
# Algorithmic Reverb (Schroeder / Freeverb design)
# ======================================================================

def _comb_filter(x: np.ndarray, delay_samples: int,
                 feedback: float, damping: float) -> np.ndarray:
    """
    Feedback comb filter with first-order low-pass damping inside the loop.

        y[n] = x[n] + feedback · damp_filter(y[n - delay])

    The damping low-pass is a one-pole IIR:
        damp_out[n] = (1-damping)·y[n-delay] + damping·prev_damp

    Implemented as a manual sample loop — no scipy.signal.
    """
    N = len(x)
    y = np.zeros(N + delay_samples, dtype=np.float64)
    damp_state = 0.0

    for n in range(N):
        # Read from comb delay line
        past = y[n]  # y is offset: y[n] corresponds to the feedback tap
        # Damping filter
        damp_out = (1.0 - damping) * past + damping * damp_state
        damp_state = damp_out
        y[n + delay_samples] += x[n] + feedback * damp_out

    return y[:N]


def _allpass_filter(x: np.ndarray, delay_samples: int,
                    feedback: float) -> np.ndarray:
    """
    Schroeder all-pass filter:

        y[n] = −x[n] + x[n-M] + feedback · y[n-M]

    Preserves frequency magnitude while diffusing phase (adds density).
    Implemented as a manual sample loop.
    """
    N = len(x)
    y = np.zeros(N, dtype=np.float64)
    buf = np.zeros(delay_samples, dtype=np.float64)
    ptr = 0

    for n in range(N):
        buf_out = buf[ptr]
        buf[ptr] = x[n] + feedback * buf_out
        y[n] = -x[n] + buf_out
        ptr = (ptr + 1) % delay_samples

    return y


def apply_reverb(
    sig: Signal,
    room_size: float = 0.5,
    damping: float = 0.5,
    wet: float = 0.33,
) -> Signal:
    """
    Algorithmic reverb using a Schroeder/Freeverb-style network:
        4 parallel feedback comb filters → 2 series all-pass filters

    Parameters
    ----------
    sig       : input Signal
    room_size : controls comb filter feedback gain (0.0–1.0)
    damping   : high-frequency damping inside comb loops (0.0–1.0)
    wet       : dry/wet mix ratio (0.0–1.0)

    Returns
    -------
    Signal — reverb-processed output
    """
    if not (0.0 <= room_size <= 1.0):
        raise ValueError(f"room_size must be in [0, 1], got {room_size}")
    if not (0.0 <= wet <= 1.0):
        raise ValueError(f"wet must be in [0, 1], got {wet}")

    fs = sig.fs
    x = sig.samples

    # Comb filter delay lengths tuned for natural room acoustics
    # (prime-number lengths scaled to sampling rate to avoid resonance artefacts)
    base = int(fs / 100.0)   # ~10 ms at 1× scale
    comb_delays = [
        int(base * 1.116),
        int(base * 1.188),
        int(base * 1.277),
        int(base * 1.356),
    ]
    comb_delays = [max(2, d) for d in comb_delays]

    # All-pass delay lengths
    ap_delays = [
        int(base * 0.556),
        int(base * 0.422),
    ]
    ap_delays = [max(2, d) for d in ap_delays]

    # Feedback gain scales with room_size
    comb_gain = 0.84 * room_size + 0.12   # range ~[0.12, 0.96]
    ap_gain   = 0.5

    # 4 parallel comb filters summed
    wet_signal = np.zeros(len(x), dtype=np.float64)
    for d in comb_delays:
        wet_signal += _comb_filter(x, d, comb_gain, damping)
    wet_signal /= len(comb_delays)

    # 2 series all-pass filters for diffusion
    for d in ap_delays:
        wet_signal = _allpass_filter(wet_signal, d, ap_gain)

    # Dry/wet mix
    mixed = (1.0 - wet) * x + wet * wet_signal
    return Signal(mixed, fs)


# ======================================================================
# Convolution Reverb (Room Acoustics)
# ======================================================================

def apply_convolution_reverb(
    sig: Signal,
    rir: Signal,
    wet: float = 0.5,
) -> Signal:
    """
    Convolution reverb: convolve the input signal with a Room Impulse
    Response (RIR) to simulate room acoustics.

    Uses `convolution.convolve_fast` (our existing scratch implementation)
    trimmed back to the original signal length.

    Parameters
    ----------
    sig : input Signal
    rir : Room Impulse Response Signal (must share fs with sig)
    wet : dry/wet mix (0 = dry, 1 = fully convolved)

    Returns
    -------
    Signal — room-acoustics-processed output, same length as input
    """
    from convolution import convolve_fast

    if sig.fs != rir.fs:
        raise ValueError(
            f"Signal fs ({sig.fs}) and RIR fs ({rir.fs}) must match."
        )

    # Normalise RIR so it doesn't blow up amplitude
    rir_samples = rir.samples.copy()
    peak = np.max(np.abs(rir_samples))
    if peak > 0:
        rir_samples /= peak
    rir_norm = Signal(rir_samples, rir.fs)

    # Full linear convolution; trim back to original length
    convolved = convolve_fast(sig, rir_norm)
    wet_samples = convolved.samples[: sig.N]

    # Dry/wet mix
    mixed = (1.0 - wet) * sig.samples + wet * wet_samples
    return Signal(mixed, sig.fs)


# ======================================================================
# Advanced Noise Removal — STFT Spectral Subtraction
# ======================================================================

def _stft_frames(x: np.ndarray, frame_len: int,
                 hop: int, window: np.ndarray) -> np.ndarray:
    """
    Compute STFT frames of x using our scratch FFT.

    Returns complex matrix of shape (num_frames, frame_len).
    """
    N = len(x)
    num_frames = 1 + (N - frame_len) // hop
    frames = np.zeros((num_frames, frame_len), dtype=np.complex128)
    for i in range(num_frames):
        start = i * hop
        chunk = x[start: start + frame_len].astype(np.float64)
        windowed = chunk * window
        frames[i] = fft(windowed.astype(np.complex128))
    return frames


def _istft_overlap_add(frames: np.ndarray, hop: int,
                       window: np.ndarray, original_N: int) -> np.ndarray:
    """
    Reconstruct a time-domain signal from STFT frames via overlap-add IFFT.
    Uses our scratch ifft_real.
    """
    frame_len = frames.shape[1]
    out = np.zeros(original_N + frame_len, dtype=np.float64)
    norm = np.zeros(original_N + frame_len, dtype=np.float64)

    for i, frame_spec in enumerate(frames):
        time_frame = ifft_real(frame_spec)
        start = i * hop
        end = start + frame_len
        out[start:end] += time_frame * window
        norm[start:end] += window ** 2

    # Normalise overlap
    norm = np.where(norm < 1e-10, 1.0, norm)
    return (out / norm)[:original_N]


def spectral_subtraction(
    sig: Signal,
    noise_start_s: float = 0.0,
    noise_end_s: float = 0.2,
    over_subtraction: float = 1.0,
    frame_ms: float = 32.0,
    overlap: float = 0.75,
) -> Signal:
    """
    STFT-based spectral subtraction noise reduction.

    Algorithm:
        1. Profile noise power spectrum from a quiet segment
           [noise_start_s, noise_end_s]
        2. Compute STFT of full signal (manual frame-by-frame FFT)
        3. Subtract scaled noise magnitude from signal magnitude
           (half-wave rectify to avoid negative values)
        4. Reconstruct via ISTFT overlap-add (manual IFFT per frame)

    Parameters
    ----------
    sig              : input Signal
    noise_start_s    : start of noise-only reference segment (seconds)
    noise_end_s      : end of noise-only reference segment (seconds)
    over_subtraction : scaling factor for noise profile (≥ 1.0;
                       larger = more aggressive subtraction)
    frame_ms         : STFT frame length in milliseconds
    overlap          : fractional frame overlap (0.5–0.875 typical)

    Returns
    -------
    Signal — spectrally subtracted output
    """
    fs = sig.fs
    x = sig.samples

    # Frame parameters — frame_len must be power of 2 for our scratch FFT
    from transforms import next_power_of_two
    raw_frame_len = int(frame_ms / 1000.0 * fs)
    frame_len = next_power_of_two(raw_frame_len)
    hop = max(1, int(frame_len * (1.0 - overlap)))

    # Hann window (length = frame_len)
    window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(frame_len) / frame_len))

    # ── 1. Noise profile ────────────────────────────────────────────
    n_start = int(noise_start_s * fs)
    n_end   = int(noise_end_s * fs)
    noise_segment = x[n_start:n_end]

    if len(noise_segment) < frame_len:
        # Not enough samples — pad to frame_len
        noise_segment = np.concatenate([
            noise_segment,
            np.zeros(frame_len - len(noise_segment))
        ])

    # Average magnitude spectrum over noise frames
    noise_frames = _stft_frames(noise_segment, frame_len, hop, window)
    noise_mag_profile = np.mean(np.abs(noise_frames), axis=0)

    # ── 2. STFT of full signal ───────────────────────────────────────
    signal_frames = _stft_frames(x, frame_len, hop, window)

    # ── 3. Spectral subtraction ──────────────────────────────────────
    cleaned_frames = np.zeros_like(signal_frames)
    for i, frame in enumerate(signal_frames):
        mag = np.abs(frame)
        phase = np.angle(frame)
        # Subtract noise profile (over-subtraction factor ≥ 1.0)
        mag_clean = mag - over_subtraction * noise_mag_profile
        # Half-wave rectify to prevent negative magnitudes
        mag_clean = np.maximum(mag_clean, 0.0)
        cleaned_frames[i] = mag_clean * np.exp(1j * phase)

    # ── 4. ISTFT reconstruction ──────────────────────────────────────
    reconstructed = _istft_overlap_add(cleaned_frames, hop, window, len(x))
    return Signal(reconstructed, fs)
