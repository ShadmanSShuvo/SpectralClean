# SpectralClean v2 — Feature Implementation Plan

Implement the 4 phases described in `plan-v2.md`: EQ, Echo/Delay, Reverb, Advanced Noise Removal, unified effect chain, Streamlit UI controls, unit tests, and benchmark updates.

All DSP code follows the critical constraint: **no `scipy.signal.*`, `numpy.fft.*`, or `scipy.fft.*`** — only manual implementations using NumPy arrays.

---

## Proposed Changes

### Phase 1 — Core DSP (`effects.py` [NEW] + `filters.py` updates)

---

#### [NEW] [effects.py](file:///Users/shuvo/Documents/Dev/SpectralClean/effects.py)

All new time/frequency-domain audio effects in one module:

**1. Equalizer (`Equalizer` class)**
- Implement 2nd-order IIR biquad filters using the standard difference equation:
  `y[n] = b0·x[n] + b1·x[n-1] + b2·x[n-2] − a1·y[n-1] − a2·y[n-2]`
- Filter types (all formulas derived from Audio EQ Cookbook — implemented manually):
  - Low-pass, High-pass, Band-pass, Low-shelf, High-shelf, Peaking
- `Equalizer` class: accepts list of `(band_hz, gain_db, filter_type)` tuples and chains biquads
- 5-band graphic EQ convenience method: `apply_graphic_eq(sig, gains_db)` mapping standard bands `[60, 230, 910, 4000, 14000]` Hz

**2. Echo/Delay (`apply_echo`)**
- Direct feedback comb filter: `y[n] = x[n] + g · y[n - M]`
- `M = int(delay_ms / 1000 * fs)` samples; `g < 1.0` enforced by dataclass validation
- Dry/wet mix parameter

**3. Algorithmic Reverb (`apply_reverb`)**
- Schroeder/Freeverb design: 4 parallel comb filters → 2 series all-pass filters
- Manually computed comb/all-pass feedback loops (pure NumPy loops)
- Configurable `room_size`, `damping`, `wet` mix
- Convolution reverb: `apply_convolution_reverb(sig, rir_sig)` using existing `convolution.convolve_fast`

**4. Advanced Noise Removal (`spectral_subtraction`)**
- STFT-based spectral subtraction implemented using `transforms.fft` / `transforms.ifft`
- Frame-based STFT (Hann window, 75% overlap) via manual FFT calls — **no scipy.signal.stft**
- Profile noise from a silent segment, subtract magnitude spectrum, reconstruct via overlap-add ISTFT
- Wiener-filter variant (optional, same framework)

---

#### [MODIFY] [filters.py](file:///Users/shuvo/Documents/Dev/SpectralClean/filters.py)

No changes needed — all new DSP goes into `effects.py`. The existing `denoise_combined` remains the hum-removal entry point.

---

### Phase 2 — Signal Pipeline Integration (`signal_core.py` / `main.py`)

#### [MODIFY] [signal_core.py](file:///Users/shuvo/Documents/Dev/SpectralClean/signal_core.py)

Add an `EffectChain` dataclass (parameter bag) with Pydantic-free validation:

```python
@dataclass
class EffectChainConfig:
    noise_removal:  bool  = False   # spectral subtraction
    eq_gains_db:    list  = None    # 5 floats or None
    echo_enabled:   bool  = False
    echo_delay_ms:  float = 200.0
    echo_feedback:  float = 0.3     # must be < 1.0
    echo_wet:       float = 0.5
    reverb_enabled: bool  = False
    room_size:      float = 0.5     # 0–1
    reverb_wet:     float = 0.3
    rir_signal:     Signal = None   # for convolution reverb
```

Validation raises `ValueError` if `echo_feedback >= 1.0`.

#### [MODIFY] [main.py](file:///Users/shuvo/Documents/Dev/SpectralClean/main.py)

Add `run_effect_chain(sig, config: EffectChainConfig) -> Signal` that applies:
`Noise Removal → Equalizer → Echo → Reverb` in order, based on config flags.

---

### Phase 3 — Streamlit UI (`app.py`)

#### [MODIFY] [app.py](file:///Users/shuvo/Documents/Dev/SpectralClean/app.py)

**New sidebar section: "🎛️ Effects Chain"** added below existing Filter Controls:

- **Equalizer section** (collapsible `st.expander`):
  - 5 sliders: `60 Hz`, `230 Hz`, `910 Hz`, `4 kHz`, `14 kHz` gain (−12 dB to +12 dB, step 1 dB)
  - Toggle checkbox to enable/bypass EQ

- **Echo / Delay section** (collapsible):
  - Checkbox enable
  - Delay Time slider (10–800 ms)
  - Feedback slider (0.0–0.9, step 0.05)
  - Wet Mix slider (0.0–1.0)

- **Reverb section** (collapsible):
  - Checkbox enable
  - Radio: Algorithmic / Convolution (RIR)
  - Room Size slider (0.0–1.0) + Wet Mix slider
  - If Convolution: `st.file_uploader` for custom `.wav` RIR file

- **Advanced Noise Removal** (new checkbox):
  - Toggle to switch from mask-based to STFT spectral subtraction

The processing pipeline block calls `run_effect_chain()` after the existing `denoise_combined`.

Audio players for the chain-processed signal (`_tmp_effects.wav`) are added.

---

### Phase 4 — Tests & Benchmark

#### [MODIFY] [test_all.py](file:///Users/shuvo/Documents/Dev/SpectralClean/test_all.py)

Add new test sections (keeping all 47 existing tests intact):

**Section: "Phase 5.1 — Equalizer"**
- Apply low-pass EQ band → energy above cutoff attenuated
- Apply high-pass band → energy below cutoff attenuated
- Gain=0 dB on all bands → signal unchanged (round-trip)

**Section: "Phase 5.2 — Echo / Delay"**
- Impulse response: echo of unit impulse at n=0 must produce peaks at 0 and M
- Amplitudes of successive echoes satisfy `g^k` decay
- `feedback >= 1.0` raises `ValueError`

**Section: "Phase 5.3 — Reverb"**
- Impulse response decays monotonically over time
- Wet=0 → output equals dry input
- Wet=1.0 → only wet signal present

**Section: "Phase 5.4 — Spectral Subtraction"**
- Silent segment profiling: subtracting a dominant sinusoidal noise significantly reduces that frequency's magnitude
- Round-trip: zero noise signal → near-zero change after subtraction

#### [MODIFY] [benchmark.py](file:///Users/shuvo/Documents/Dev/SpectralClean/benchmark.py)

Add `run_effects_benchmark(block_sizes)` that times each new effect at multiple block sizes (`512`, `1024`, `4096`, `16384` samples) and adds a second sub-plot to the benchmark figure showing processing overhead vs block size.

---

## Verification Plan

### Automated Tests
```bash
python test_all.py    # all 47 existing + ~18 new tests must pass
python main.py        # pipeline must complete, output metrics + report/ files
```

### Manual Verification
- `streamlit run app.py` → new sidebar sections appear and controls update audio
- Toggle EQ, Echo, Reverb — confirm audio changes audibly
- Upload a RIR `.wav` → convolution reverb applied

---

## Key Design Decisions

> [!IMPORTANT]
> **`effects.py` uses only NumPy** — no `scipy.signal.lfilter`. The biquad IIR is implemented as a manual sample-by-sample loop satisfying the grader constraint.

> [!IMPORTANT]
> **STFT in `spectral_subtraction`** uses `transforms.fft` (our scratch FFT) inside each frame window — no `scipy.signal.stft`.

> [!NOTE]
> `EffectChainConfig` uses a plain Python dataclass (not Pydantic) to avoid new dependencies. Validation is a simple `__post_init__`.
