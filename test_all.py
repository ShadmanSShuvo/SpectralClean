"""
test_all.py — Phase 1 & 2 unit tests
Validates every primitive against hand-calculated expected outputs
and cross-checks FFT against numpy (in test mode only).

Run: python test_all.py
"""

import sys
import numpy as np

# ── Colour helpers ─────────────────────────────────────────────────────
GREEN = "\033[92m"
RED   = "\033[91m"
CYAN  = "\033[96m"
BOLD  = "\033[1m"
RESET = "\033[0m"

PASS = f"{GREEN}PASS{RESET}"
FAIL = f"{RED}FAIL{RESET}"

_failures = []

def check(name: str, condition: bool, detail: str = ""):
    if condition:
        print(f"  {PASS}  {name}")
    else:
        tag = f"{FAIL}  {name}"
        if detail:
            tag += f"  ({detail})"
        print(f"  {tag}")
        _failures.append(name)


def section(title: str):
    print(f"\n{BOLD}{CYAN}── {title} {'─' * max(0, 54 - len(title))}{RESET}")


# ══════════════════════════════════════════════════════════════════════
#  Phase 1.1 — Signal class
# ══════════════════════════════════════════════════════════════════════

section("Phase 1.1 — Signal representation")
from signal_core import Signal, unit_impulse, sine_wave, white_noise

s = Signal([1, 2, 3, 4, 5], fs=10.0)
check("N == 5",          s.N == 5)
check("fs == 10.0",      s.fs == 10.0)
check("duration == 0.5", abs(s.duration - 0.5) < 1e-12)
check("t[0] == 0.0",     abs(s.t[0]) < 1e-12)
check("t[-1] == 0.4",    abs(s.t[-1] - 0.4) < 1e-10)
check("dtype float64",   s.samples.dtype == np.float64)

try:
    bad = Signal([1, 2], fs=0)
    check("ValueError on fs=0", False, "no exception raised")
except ValueError:
    check("ValueError on fs=0", True)


# ══════════════════════════════════════════════════════════════════════
#  Phase 1.2 — Elementary operations
# ══════════════════════════════════════════════════════════════════════

section("Phase 1.2 — Elementary operations")

s = Signal([1.0, 2.0, 3.0, 4.0, 5.0], fs=10.0)

# Amplitude scale
scaled = s.scale(3.0)
check("scale: samples × 3", np.allclose(scaled.samples, [3, 6, 9, 12, 15]))

# Shift right by 2
shifted = s.shift(2)
check("shift(+2): first 2 zeros", np.allclose(shifted.samples[:2], 0))
check("shift(+2): samples[2] == 1", abs(shifted.samples[2] - 1.0) < 1e-12)
check("shift(+2): samples[4] == 3", abs(shifted.samples[4] - 3.0) < 1e-12)
check("shift(+2): length preserved", shifted.N == s.N)

# Shift left by 1
shifted_l = s.shift(-1)
check("shift(-1): samples[0] == 2", abs(shifted_l.samples[0] - 2.0) < 1e-12)
check("shift(-1): last sample == 0", abs(shifted_l.samples[-1]) < 1e-12)

# Add — same length
a = Signal([1.0, 2.0, 3.0], fs=8.0)
b = Signal([4.0, 5.0, 6.0], fs=8.0)
added = a.add(b)
check("add: element-wise sum", np.allclose(added.samples, [5, 7, 9]))

# Add — length mismatch (zero-padding)
c = Signal([1.0, 2.0], fs=8.0)
d = Signal([10.0, 20.0, 30.0], fs=8.0)
added2 = c.add(d)
check("add (mismatch): length = max", added2.N == 3)
check("add (mismatch): [11, 22, 30]", np.allclose(added2.samples, [11, 22, 30]))

# Multiply
m = a.multiply(b)
check("multiply: element-wise product", np.allclose(m.samples, [4, 10, 18]))

# fs mismatch guard
try:
    Signal([1], 8.0).add(Signal([1], 16.0))
    check("fs mismatch raises ValueError", False, "no exception")
except ValueError:
    check("fs mismatch raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  Phase 1.3 — Manual convolution
# ══════════════════════════════════════════════════════════════════════

section("Phase 1.3 — Manual convolution")
from convolution import convolve, convolve_fast, moving_average_kernel

# Impulse response test: convolving impulse with arbitrary h should give h
fs = 100.0
h_arr = np.array([0.5, 1.0, -0.5, 0.25])
impulse = Signal(np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), fs)
h_sig   = Signal(h_arr, fs)
conv_out = convolve(impulse, h_sig)
# First Nh samples should equal h_arr
check("convolve: impulse response = h",
      np.allclose(conv_out.samples[:len(h_arr)], h_arr, atol=1e-12))

# Moving average smoothing on a constant signal [1,1,...,1]
const = Signal(np.ones(20), fs)
avg = moving_average_kernel(5, fs)
smoothed = convolve(const, avg)
# Interior of moving-average of all-ones should still be 1
check("moving-avg of all-ones = 1 (interior)",
      np.allclose(smoothed.samples[4:18], 1.0, atol=1e-12))

# Cross-check convolve vs convolve_fast
x = Signal(np.array([1.0, 2.0, -1.0, 3.0]), fs)
k = Signal(np.array([0.5, 0.5]), fs)
ref  = convolve(x, k)
fast = convolve_fast(x, k)
check("convolve vs convolve_fast: identical output",
      np.allclose(ref.samples, fast.samples, atol=1e-12))

# Cross-check against numpy.convolve (test-only)
np_ref = np.convolve(x.samples, k.samples, mode="full")
check("convolve matches np.convolve",
      np.allclose(ref.samples, np_ref, atol=1e-12))

# Output length = Nx + Nh - 1
check("convolve: output length = Nx+Nh-1",
      ref.N == len(x.samples) + len(k.samples) - 1)


# ══════════════════════════════════════════════════════════════════════
#  Phase 2.1 — Naive DFT
# ══════════════════════════════════════════════════════════════════════

section("Phase 2.1 — Naive DFT")
from transforms import dft, dft_slow, next_power_of_two, zero_pad_to_power2

# DC-only signal: all samples equal → X[0] = N, X[k≠0] = 0
N = 8
x_dc = np.ones(N, dtype=np.complex128)
X_dc = dft(x_dc)
check("DFT: DC input → X[0] = N", abs(X_dc[0] - N) < 1e-10)
check("DFT: DC input → X[k≠0] ≈ 0", np.all(np.abs(X_dc[1:]) < 1e-10))

# Single tone: x[n] = e^{j 2π k₀ n / N} → X[k₀] = N, X[k≠k₀] = 0
k0 = 3
x_tone = np.exp(2j * np.pi * k0 * np.arange(N) / N)
X_tone = dft(x_tone)
check("DFT: pure tone k₀=3 → X[3] = N",  abs(X_tone[k0] - N) < 1e-8)
check("DFT: pure tone k₀=3 → others ≈ 0",
      all(abs(X_tone[k]) < 1e-8 for k in range(N) if k != k0))

# Matrix vs nested-loop agree
rng = np.random.default_rng(7)
x_rand = rng.standard_normal(16).astype(np.complex128)
X_vec   = dft(x_rand)
X_loop  = dft_slow(x_rand)
check("DFT vectorised == DFT nested-loop",
      np.allclose(X_vec, X_loop, atol=1e-10))


# ══════════════════════════════════════════════════════════════════════
#  Phase 2.2 & 2.3 — FFT and IFFT
# ══════════════════════════════════════════════════════════════════════

section("Phase 2.2 — Cooley-Tukey FFT")
from transforms import fft, ifft, ifft_real, validate_against_numpy

# Power-of-2 constraint
try:
    fft(np.ones(7))
    check("FFT raises on non-power-of-2", False, "no exception")
except ValueError:
    check("FFT raises on non-power-of-2", True)

# N=1 base case
check("FFT: N=1 returns same value", np.allclose(fft(np.array([5+0j])), [5+0j]))

# Linearity
N = 64
x1 = rng.standard_normal(N).astype(np.complex128)
x2 = rng.standard_normal(N).astype(np.complex128)
a, b = 2.3, -1.7
check("FFT linearity: F(ax+by) = aF(x)+bF(y)",
      np.allclose(fft(a * x1 + b * x2),
                  a * fft(x1) + b * fft(x2), atol=1e-9))

# Cross-check against numpy.fft
result = validate_against_numpy(rng.standard_normal(512))
check(f"FFT matches numpy.fft (max err={result['fft_max_error']:.2e})",
      result["fft_pass"])
check(f"DFT matches numpy.fft (max err={result['dft_max_error']:.2e})",
      result["dft_pass"])

section("Phase 2.3 — IFFT (conjugate trick)")
# Round-trip: IFFT(FFT(x)) ≈ x
N = 128
x_orig = rng.standard_normal(N).astype(np.complex128)
x_rt   = ifft(fft(x_orig))
check("IFFT∘FFT round-trip (complex)",
      np.allclose(x_rt, x_orig, atol=1e-10))

# Real input → real output
x_real = rng.standard_normal(64)
x_padded = zero_pad_to_power2(x_real)
rec = ifft_real(fft(x_padded))
check("IFFT of real FFT → real values",
      np.allclose(rec[:len(x_real)], x_real, atol=1e-10))

# Parseval's theorem: power preserved
N = 256
x_p = rng.standard_normal(N).astype(np.complex128)
X_p = fft(x_p)
power_time = np.sum(np.abs(x_p) ** 2)
power_freq = np.sum(np.abs(X_p) ** 2) / N
check("Parseval's theorem: time power ≈ freq power / N",
      abs(power_time - power_freq) / power_time < 1e-8)


# ══════════════════════════════════════════════════════════════════════
#  Phase 4.1 — SNR metrics
# ══════════════════════════════════════════════════════════════════════

section("Phase 4.1 — SNR metrics")
from metrics import snr_db, psnr_db, rmse, snr_improvement

fs_m = 100.0
t_m  = np.arange(100) / fs_m
clean_m = Signal(np.sin(2 * np.pi * 5 * t_m), fs_m)  # 5 Hz sine

# Noiseless: SNR should be +inf
check("SNR(clean, clean) = +inf",
      snr_db(clean_m, clean_m) == float("inf"))

# Known noise level
noise_std = 0.1
noisy_m   = Signal(clean_m.samples + noise_std * rng.standard_normal(100), fs_m)
snr_val   = snr_db(clean_m, noisy_m)
check("SNR(clean, noisy) < SNR(clean, clean)",
      snr_val < float("inf"))
check("SNR roughly consistent with σ=0.1",
      0 < snr_val < 40)   # 5 Hz sine power ≈ 0.5; noise power ≈ 0.01 → ~17 dB

# RMSE = 0 for identical signals
check("RMSE(identical) == 0", rmse(clean_m, clean_m) == 0.0)

# Improvement: denoised closer to clean than noisy
# simulate a "denoised" signal with less noise
denoised_m = Signal(clean_m.samples + 0.01 * rng.standard_normal(100), fs_m)
report = snr_improvement(clean_m, noisy_m, denoised_m)
check("SNR improvement > 0 when denoised < noisy",
      report["improvement_db"] > 0)


# ══════════════════════════════════════════════════════════════════════
#  Phase 4.3 — Edge cases
# ══════════════════════════════════════════════════════════════════════

section("Phase 4.3 — Edge cases")
from filters import denoise_combined, apply_spectral_filter, band_stop_mask

# Silence → stays silent (or near-silent after denoising)
silence = Signal(np.zeros(512), 8000.0)
sil_den = denoise_combined(silence, hum_freq_hz=50)
check("Silence through denoiser → near-zero",
      np.max(np.abs(sil_den.samples)) < 1e-10)

# Very short signal (N=4)
short = Signal(np.array([1.0, -1.0, 1.0, -1.0]), 8000.0)
try:
    short_den = denoise_combined(short, hum_freq_hz=50)
    check("Very short signal (N=4) processes without crash", True)
except Exception as e:
    check("Very short signal (N=4) processes without crash", False, str(e))

# Clipped signal (samples ±1)
clipped = Signal(np.clip(2.0 * np.sin(2 * np.pi * 200 * np.arange(512) / 8000), -1, 1), 8000.0)
try:
    clip_den = denoise_combined(clipped, hum_freq_hz=50)
    check("Clipped signal processes without crash", True)
except Exception as e:
    check("Clipped signal processes without crash", False, str(e))

# Mask length mismatch guard
wrong_mask = np.ones(128)
x_pad = zero_pad_to_power2(Signal(np.ones(500), 8000.0).samples)
try:
    apply_spectral_filter(Signal(np.ones(500), 8000.0), wrong_mask)
    check("Mask length mismatch raises ValueError", False, "no exception")
except ValueError:
    check("Mask length mismatch raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  Phase 5.1 — Equalizer (IIR biquad)
# ══════════════════════════════════════════════════════════════════════

section("Phase 5.1 — Equalizer (IIR Biquad)")
from effects import Equalizer, EQ_BANDS_HZ

fs_eq = 8000.0

# Flat EQ (all 0 dB gains) → signal unchanged
flat_sig = Signal(np.sin(2 * np.pi * 440 * np.arange(512) / fs_eq), fs_eq)
eq_flat = Equalizer.graphic_eq([0.0, 0.0, 0.0, 0.0, 0.0], fs_eq)
out_flat = eq_flat.apply(flat_sig)
check("EQ flat (0 dB all bands): output length preserved", out_flat.N == flat_sig.N)
check("EQ flat (0 dB all bands): samples near-identical",
      np.allclose(out_flat.samples, flat_sig.samples, atol=1e-6))

# Peaking boost at 440 Hz → magnitude at 440 Hz increases
t_eq = np.arange(2048) / fs_eq
x_440 = np.sin(2 * np.pi * 440 * t_eq)
sig_440 = Signal(x_440, fs_eq)
eq_boost = Equalizer([('peaking', 440.0, +12.0, 1.0)], fs_eq)
out_boost = eq_boost.apply(sig_440)
check("EQ peaking +12 dB at 440 Hz: output RMS > input RMS",
      np.sqrt(np.mean(out_boost.samples**2)) > np.sqrt(np.mean(x_440**2)))

# Peaking cut at 440 Hz → magnitude decreases
eq_cut = Equalizer([('peaking', 440.0, -12.0, 1.0)], fs_eq)
out_cut = eq_cut.apply(sig_440)
check("EQ peaking −12 dB at 440 Hz: output RMS < input RMS",
      np.sqrt(np.mean(out_cut.samples**2)) < np.sqrt(np.mean(x_440**2)))

# Wrong number of gains raises ValueError
try:
    Equalizer.graphic_eq([0.0, 0.0], fs_eq)
    check("EQ: wrong gains count raises ValueError", False, "no exception")
except ValueError:
    check("EQ: wrong gains count raises ValueError", True)

# Low-pass type: energy above cutoff attenuated
freqs_above = np.sin(2 * np.pi * 3000 * np.arange(2048) / fs_eq)
sig_hi = Signal(freqs_above, fs_eq)
eq_lp = Equalizer([('lowpass', 500.0, 0.0, 0.707)], fs_eq)
out_lp = eq_lp.apply(sig_hi)
check("EQ lowpass: high-freq signal energy reduced",
      np.sqrt(np.mean(out_lp.samples**2)) < np.sqrt(np.mean(freqs_above**2)) * 0.5)


# ══════════════════════════════════════════════════════════════════════
#  Phase 5.2 — Echo / Delay
# ══════════════════════════════════════════════════════════════════════

section("Phase 5.2 — Echo / Delay")
from effects import apply_echo

fs_echo = 8000.0
# Impulse response test: unit impulse at n=0 → echoes at M, 2M, …
M = int(0.1 * fs_echo)   # 100 ms = 800 samples
impulse_sig = Signal(np.concatenate([[1.0], np.zeros(5 * M - 1)]), fs_echo)
g = 0.5
echo_out = apply_echo(impulse_sig, delay_ms=100.0, feedback=g, wet=1.0)
# At wet=1: mixed = 0*dry + 1*wet = wet signal
# wet signal y[n] = x[n] + g*y[n-M], so y[0]=1, y[M]=g, y[2M]=g^2, ...
# But mixed = (1-wet)*dry + wet*wet_y = wet_y when wet=1
# Check: peak at sample 0 exists
check("Echo impulse: peak at sample 0 exists",
      abs(echo_out.samples[0]) > 0.4)

# First echo at M samples: amplitude ≈ g  (within tolerance for comb filter)
check("Echo impulse: first echo at M samples present",
      abs(echo_out.samples[M]) > 0.1)

# Successive echoes decay (y[2M] < y[M] in magnitude)
check("Echo impulse: echoes decay (2nd < 1st)",
      abs(echo_out.samples[2 * M]) <= abs(echo_out.samples[M]) + 1e-6)

# feedback ≥ 1.0 raises ValueError
try:
    apply_echo(impulse_sig, feedback=1.0)
    check("Echo: feedback=1.0 raises ValueError", False, "no exception")
except ValueError:
    check("Echo: feedback=1.0 raises ValueError", True)

# wet=0 → output equals dry input exactly
dry_only = apply_echo(impulse_sig, delay_ms=100.0, feedback=0.5, wet=0.0)
check("Echo: wet=0 → output equals input",
      np.allclose(dry_only.samples, impulse_sig.samples, atol=1e-10))


# ══════════════════════════════════════════════════════════════════════
#  Phase 5.3 — Reverb (Algorithmic + Convolution)
# ══════════════════════════════════════════════════════════════════════

section("Phase 5.3 — Reverb")
from effects import apply_reverb, apply_convolution_reverb

fs_rv = 8000.0
imp_rv = Signal(np.concatenate([[1.0], np.zeros(4000 - 1)]), fs_rv)

# Dry-only: wet=0 → output = input
dry_rv = apply_reverb(imp_rv, room_size=0.5, wet=0.0)
check("Reverb: wet=0 → output equals input",
      np.allclose(dry_rv.samples, imp_rv.samples, atol=1e-10))

# With reverb: output longer-tailed than input (RMS over tail > 0)
wet_rv = apply_reverb(imp_rv, room_size=0.7, wet=0.8)
tail_energy = np.sqrt(np.mean(wet_rv.samples[200:]**2))
check("Reverb: tail energy > 0 after reverb applied",
      tail_energy > 1e-4)

# room_size out of range raises ValueError
try:
    apply_reverb(imp_rv, room_size=1.5)
    check("Reverb: room_size > 1 raises ValueError", False, "no exception")
except ValueError:
    check("Reverb: room_size > 1 raises ValueError", True)

# Convolution reverb: using a simple exponential decay as "RIR"
t_rir = np.arange(400) / fs_rv
rir_samples = np.exp(-10 * t_rir) * np.sin(2 * np.pi * 300 * t_rir)
rir_sig = Signal(rir_samples, fs_rv)
sine_in = Signal(np.sin(2 * np.pi * 300 * np.arange(800) / fs_rv), fs_rv)
conv_rv = apply_convolution_reverb(sine_in, rir_sig, wet=0.5)
check("Convolution reverb: output same length as input",
      conv_rv.N == sine_in.N)
check("Convolution reverb: output different from dry input",
      not np.allclose(conv_rv.samples, sine_in.samples, atol=1e-6))

# fs mismatch raises ValueError
rir_wrong_fs = Signal(rir_samples, fs_rv * 2)
try:
    apply_convolution_reverb(sine_in, rir_wrong_fs, wet=0.5)
    check("Convolution reverb: fs mismatch raises ValueError", False, "no exception")
except ValueError:
    check("Convolution reverb: fs mismatch raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  Phase 5.4 — Spectral Subtraction
# ══════════════════════════════════════════════════════════════════════

section("Phase 5.4 — Spectral Subtraction")
from effects import spectral_subtraction

fs_ss = 8000.0
duration_ss = 1.0
N_ss = int(fs_ss * duration_ss)
t_ss = np.arange(N_ss) / fs_ss

# Build signal: silent noise prefix + clean tone
noise_prefix = 0.2 * np.sin(2 * np.pi * 300 * t_ss[:int(0.2 * fs_ss)])  # dominant noise at 300 Hz
signal_part  = np.sin(2 * np.pi * 1000 * t_ss[int(0.2 * fs_ss):])        # speech tone at 1 kHz
noisy_ss = Signal(np.concatenate([noise_prefix, signal_part]), fs_ss)

ss_out = spectral_subtraction(
    noisy_ss,
    noise_start_s=0.0,
    noise_end_s=0.2,
    over_subtraction=2.0,
)
check("Spectral subtraction: output same length as input", ss_out.N == noisy_ss.N)

# Energy in the first 0.2 s (noise region) should be reduced
noise_region_in  = np.sqrt(np.mean(noisy_ss.samples[:int(0.2 * fs_ss)]**2))
noise_region_out = np.sqrt(np.mean(ss_out.samples[:int(0.2 * fs_ss)]**2))
check("Spectral subtraction: noise region energy reduced",
      noise_region_out < noise_region_in)

# Silent input → output stays near-zero
silence_ss = Signal(np.zeros(N_ss), fs_ss)
ss_sil = spectral_subtraction(silence_ss)
check("Spectral subtraction: silence in → silence out",
      np.max(np.abs(ss_sil.samples)) < 1e-8)


# ══════════════════════════════════════════════════════════════════════
#  Phase 5.5 — EffectChainConfig validation
# ══════════════════════════════════════════════════════════════════════

section("Phase 5.5 — EffectChainConfig validation")
from signal_core import EffectChainConfig

# Default config is valid (no exceptions)
try:
    cfg = EffectChainConfig()
    check("EffectChainConfig: default config valid", True)
except Exception as e:
    check("EffectChainConfig: default config valid", False, str(e))

# echo_feedback >= 1.0 raises ValueError
try:
    EffectChainConfig(echo_feedback=1.0)
    check("EffectChainConfig: feedback=1.0 raises ValueError", False, "no exception")
except ValueError:
    check("EffectChainConfig: feedback=1.0 raises ValueError", True)

# reverb_room_size > 1 raises ValueError
try:
    EffectChainConfig(reverb_room_size=1.5)
    check("EffectChainConfig: room_size>1 raises ValueError", False, "no exception")
except ValueError:
    check("EffectChainConfig: room_size>1 raises ValueError", True)

# eq_gains_db wrong length raises ValueError
try:
    EffectChainConfig(eq_gains_db=[0.0, 0.0])
    check("EffectChainConfig: wrong EQ gains length raises ValueError", False, "no exception")
except ValueError:
    check("EffectChainConfig: wrong EQ gains length raises ValueError", True)


# ══════════════════════════════════════════════════════════════════════
#  Final report
# ══════════════════════════════════════════════════════════════════════

n_fail = len(_failures)
n_pass = len([x for x in [True] * 100])   # just use failure count

print(f"\n{'═' * 56}")
if n_fail == 0:
    print(f"{GREEN}{BOLD}  ✓  All tests passed!{RESET}")
else:
    print(f"{RED}{BOLD}  ✗  {n_fail} test(s) FAILED:{RESET}")
    for f in _failures:
        print(f"      • {f}")
print(f"{'═' * 56}\n")

sys.exit(0 if n_fail == 0 else 1)
