### Phase 1: Core DSP Implementation (`filters.py` & New Modules)

The backend agent will focus on adding the mathematical algorithms required for each audio effect inside `filters.py` or a newly created `effects.py` module.

* **Task 1: Equalizers (EQ)**
* *Implementation:* Build a parametric/graphic equalizer using standard IIR biquad filters (Low-pass, High-pass, Peaking, Band-pass, Low-shelf, High-shelf).
* *Agent Action:* Add an `Equalizer` class implementing standard difference equations using `scipy.signal.lfilter` or direct NumPy array operations.


* **Task 2: Add Echo & Delay**
* *Implementation:* Time-domain circular buffer delay lines with customizable feedback gain and delay time parameters.
* *Formula:* $y[n] = x[n] + g \cdot y[n - M]$ where $g$ is feedback and $M$ is the delay sample count.


* **Task 3: Reverberation & Room Acoustics Simulation**
* *Implementation:* Dual approach:
1. *Algorithmic:* Implement a simple Schroeder or Freeverb design using parallel comb filters followed by series all-pass filters.
2. *Convolution Reverb (Room Acoustics):* Leverage the existing `convolution.py` codebase to convolve input audio with an uploaded/selected Room Impulse Response (RIR) file.




* **Task 4: Advanced Noise Removal**
* *Implementation:* Spectral subtraction or Wiener filtering using the Short-Time Fourier Transform (STFT) inside `transforms.py`.
* *Agent Action:* Profile noise during silent periods, compute the noise profile magnitude spectrum, and subtract it from the audio magnitude spectrum while tracking phase.



---

### Phase 2: Signal Pipeline Integration (`signal_core.py` / `main.py`)

This agent ensures that the effects are linked correctly within the execution engine.

* **Task 1: Unified Effect Chain**
* Update the core processing function to accept a configuration dictionary outlining the sequence of processing nodes:
`Noise Removal -> Equalizer -> Echo -> Reverb`.


* **Task 2: Parameter Verification**
* Implement data validation schemas (e.g., using `pydantic` or simple dataclass checks) to ensure safety boundaries (e.g., keeping feedback gain $g < 1.0$ to prevent dangerous audio feedback loops).



---

### Phase 3: Web UI Integration (`app.py`)

The frontend agent will create control mechanisms for users to interact with these features dynamically.

* **Task 1: Add Configuration Controls**
* Add layout sliders, toggles, and dropdown menus for the new functions inside your Streamlit/Gradio web application UI (`app.py`).


* **Task 2: Feature-Specific UI Widgets**
* *Equalizer:* A multi-slider dashboard mapping standard frequency bands (e.g., 60Hz, 230Hz, 910Hz, 4kHz, 14kHz).
* *Echo/Reverb:* Sliders for Mix (Dry/Wet ratio), Delay Time (ms), Feedback, and Room Size.
* *Acoustics:* A file uploader component enabling custom WAV Room Impulse Response (RIR) profile uploads.



---

### Phase 4: Verification & Performance Evaluation (`test_all.py` & `benchmark.py`)

The testing agent validates accuracy and guards against degradation.

* **Task 1: Unit Regression Testing**
* Add test matrices to `test_all.py` validating that impulse responses output expected decay properties (e.g., verifying that an echo function generates diminishing amplitudes over predictable windows).


* **Task 2: Performance Profiling**
* Update `benchmark.py` to evaluate the processing overhead added by these time-domain and frequency-domain additions, ensuring audio rendering scales effectively with larger block sizes.